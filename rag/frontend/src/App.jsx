import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Users, FileText, Search, Menu, X, Plus, Download, MessageSquare, Trash2, LogIn, LogOut } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Auth from './Auth';
import { supabase, authEnabled, getAccessToken } from './supabaseClient';

const API_BASE = '/api';

const NARA_BASE_URL = 'https://storage.googleapis.com/jfkweb-prod';

function getNaraUrl(filename) {
  const fileId = filename.replace(/\.pdf$/i, '');
  return `${NARA_BASE_URL}/${encodeURIComponent(fileId)}.pdf`;
}

// Inject clickable citation links into markdown text
function injectCitationLinks(text, sources) {
  if (!sources || sources.length === 0) return text;

  // Replace [1], [2][3], etc. with markdown links
  return text.replace(/\[(\d+)\]/g, (match, num) => {
    const idx = parseInt(num, 10) - 1;
    if (idx >= 0 && idx < sources.length) {
      const s = sources[idx];
      const url = getNaraUrl(s.filename) + `#page=${s.page}`;
      const title = s.filename.replace(/"/g, '\\"');
      return `[\\[${num}\\]](${url} "${title}, p. ${s.page}")`;
    }
    return match;
  });
}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function chatTitle(messages) {
  const first = messages.find(m => m.role === 'user');
  if (!first) return 'New Chat';
  const text = first.content;
  return text.length > 40 ? text.slice(0, 40) + '...' : text;
}

// Inline markdown -> HTML. Input is already HTML-escaped, so quotes inside
// link titles arrive as &quot; and are matched as such.
function mdInline(escaped) {
  return escaped
    // Code spans first, so ** and * inside them are left alone.
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // [text](url "title") — link text may contain backslash-escaped brackets,
    // which is exactly what injectCitationLinks emits for [\[1\]].
    .replace(
      /\[((?:\\.|[^\]\\])*)\]\(([^)\s]+)(?:\s+&quot;(.*?)&quot;)?\)/g,
      (_m, text, url, title) =>
        `<a href="${url}"${title ? ` title="${title}"` : ''}>${text}</a>`
    )
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    // Finally drop markdown backslash escapes (\[1\] -> [1]).
    .replace(/\\([\\`*_{}[\]()#+\-.!])/g, '$1');
}

// Minimal Markdown -> HTML renderer for the PDF export.
//
// This previously called renderToStaticMarkup(<ReactMarkdown/>), which pulled
// React's entire server renderer into the *client* bundle purely to build an
// export document — bytes every visitor downloads for a feature most never
// use. The export only ever sees LLM-generated markdown (headings, bold,
// lists, tables, and the citation links injected above), so a string transform
// covers it at a fraction of the download cost.
function renderMarkdownToHTML(text) {
  const lines = escapeHtml(text).replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let listType = null;
  let i = 0;

  const closeList = () => {
    if (listType) { out.push(`</${listType}>`); listType = null; }
  };

  while (i < lines.length) {
    const line = lines[i];

    if (/^\s*```/.test(line)) {                       // fenced code block
      closeList();
      const body = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i++;
      out.push(`<pre><code>${body.join('\n')}</code></pre>`);
      continue;
    }

    // GFM table: header row followed by a |---|---| separator
    if (line.includes('|') && i + 1 < lines.length &&
        /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
      closeList();
      const cells = (row) => row.replace(/^\s*\|/, '').replace(/\|\s*$/, '')
        .split('|').map((c) => mdInline(c.trim()));
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        rows.push(cells(lines[i++]));
      }
      out.push(
        '<table><thead><tr>' + head.map((c) => `<th>${c}</th>`).join('') +
        '</tr></thead><tbody>' +
        rows.map((r) => '<tr>' + r.map((c) => `<td>${c}</td>`).join('') + '</tr>').join('') +
        '</tbody></table>'
      );
      continue;
    }

    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      out.push(`<h${level}>${mdInline(heading[2].trim())}</h${level}>`);
      i++; continue;
    }

    if (/^\s*([-*_])\s*(\1\s*){2,}$/.test(line)) {    // horizontal rule
      closeList(); out.push('<hr/>'); i++; continue;
    }

    const quote = line.match(/^\s*&gt;\s?(.*)$/);      // '>' is escaped by now
    if (quote) {
      closeList();
      out.push(`<blockquote>${mdInline(quote[1])}</blockquote>`);
      i++; continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const want = bullet ? 'ul' : 'ol';
      if (listType !== want) { closeList(); out.push(`<${want}>`); listType = want; }
      out.push(`<li>${mdInline((bullet || numbered)[1])}</li>`);
      i++; continue;
    }

    if (!line.trim()) { closeList(); i++; continue; }

    closeList();                                       // paragraph
    const para = [];
    while (i < lines.length && lines[i].trim() &&
           !/^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|```|&gt;)/.test(lines[i])) {
      para.push(lines[i++]);
    }
    out.push(`<p>${mdInline(para.join(' '))}</p>`);
  }

  closeList();
  return out.join('\n');
}

function downloadChat(messages) {
  const exportedAt = new Date().toLocaleString();
  const sections = messages.map((msg) => {
    if (msg.role === 'user') {
      return `<section class="turn user"><h2>Q: ${escapeHtml(msg.content)}</h2></section>`;
    }
    const linked = injectCitationLinks(msg.content, msg.sources);
    let html = `<section class="turn assistant">${renderMarkdownToHTML(linked)}`;
    if (msg.sources && msg.sources.length > 0) {
      html += '<div class="sources"><h3>Sources</h3><ol>';
      msg.sources.forEach((s) => {
        const url = getNaraUrl(s.filename) + `#page=${s.page}`;
        html += `<li><a href="${url}">${escapeHtml(s.filename)}, p. ${s.page}</a></li>`;
      });
      html += '</ol></div>';
    }
    html += '</section>';
    return html;
  }).join('\n');

  const doc = `<!doctype html>
<html><head><meta charset="utf-8"/>
<title>JFK Files Research — Chat Export</title>
<style>
  @page { margin: 18mm; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: #111; line-height: 1.55; max-width: 760px; margin: 0 auto; padding: 1.5rem; }
  h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
  .meta { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: 0.75rem; }
  .turn { margin-bottom: 1.25rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; page-break-inside: avoid; }
  .turn.user h2 { font-size: 1.05rem; color: #1a3a6c; margin: 0 0 0.5rem; }
  .turn.assistant { font-size: 0.95rem; }
  .turn.assistant h1, .turn.assistant h2, .turn.assistant h3 { color: #222; }
  .turn.assistant h3 { font-size: 1rem; margin-top: 1rem; }
  .turn.assistant p { margin: 0.5rem 0; }
  .turn.assistant ul, .turn.assistant ol { margin: 0.5rem 0 0.5rem 1.25rem; }
  .turn.assistant code { background: #f4f4f4; padding: 0 0.2em; border-radius: 3px; font-size: 0.9em; }
  .turn.assistant pre { background: #f4f4f4; padding: 0.75rem; border-radius: 4px; overflow-x: auto; }
  .turn.assistant a { color: #1a5fb4; text-decoration: none; }
  .turn.assistant a:hover { text-decoration: underline; }
  .turn.assistant blockquote { margin: 0.5rem 0; padding-left: 0.9rem; border-left: 3px solid #ddd; color: #444; }
  .turn.assistant table { border-collapse: collapse; width: 100%; margin: 0.75rem 0; font-size: 0.9em; }
  .turn.assistant th, .turn.assistant td { border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left; }
  .turn.assistant th { background: #f4f4f4; }
  .sources { margin-top: 0.75rem; font-size: 0.85rem; }
  .sources h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: #666; margin: 0 0 0.4rem; }
  .sources ol { margin: 0 0 0 1.2rem; }
  .sources a { color: #1a5fb4; word-break: break-all; }
</style>
</head><body>
<h1>JFK Files Research — Chat Export</h1>
<div class="meta">Exported: ${escapeHtml(exportedAt)}</div>
${sections}
<script>window.addEventListener('load', () => { setTimeout(() => window.print(), 250); });</script>
</body></html>`;

  const w = window.open('', '_blank');
  if (!w) {
    alert('Pop-up blocked. Please allow pop-ups to export PDF.');
    return;
  }
  w.document.open();
  w.document.write(doc);
  w.document.close();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function App() {
  const [chats, setChats] = useState([{ id: generateId(), messages: [] }]);
  const [activeChatId, setActiveChatId] = useState(chats[0].id);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [session, setSession] = useState(null);
  const [authOpen, setAuthOpen] = useState(false);
  const [authReason, setAuthReason] = useState('');
  const [quotaLeft, setQuotaLeft] = useState(null);
  const chatEndRef = useRef(null);

  const activeChat = chats.find(c => c.id === activeChatId) || chats[0];
  const messages = activeChat.messages;

  const setMessages = (updater) => {
    setChats(prev => prev.map(c =>
      c.id === activeChatId
        ? { ...c, messages: typeof updater === 'function' ? updater(c.messages) : updater }
        : c
    ));
  };

  const createNewChat = () => {
    const newChat = { id: generateId(), messages: [] };
    setChats(prev => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    setSidebarOpen(false);
  };

  const deleteChat = (id, e) => {
    e.stopPropagation();
    if (chats.length === 1) {
      // Last chat — just clear it
      setChats([{ id: generateId(), messages: [] }]);
      setActiveChatId(chats[0]?.id);
      return;
    }
    const remaining = chats.filter(c => c.id !== id);
    setChats(remaining);
    if (activeChatId === id) {
      setActiveChatId(remaining[0].id);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  // Track the Supabase session. detectSessionInUrl handles the magic-link
  // callback, which fires onAuthStateChange rather than a page-load event.
  useEffect(() => {
    if (!authEnabled) return;
    supabase.auth.getSession().then(({ data }) => setSession(data?.session ?? null));
    const { data: sub } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next);
      if (next) {
        // Signing in clears the trial banner and closes the modal.
        setAuthOpen(false);
        setQuotaLeft(null);
        // Strip the magic-link tokens from the address bar.
        window.history.replaceState({}, document.title, window.location.pathname);
      }
    });
    return () => sub?.subscription?.unsubscribe();
  }, []);

  const promptSignIn = (reason) => {
    setAuthReason(reason || '');
    setAuthOpen(true);
  };

  const signOut = async () => {
    if (authEnabled) await supabase.auth.signOut();
    setSession(null);
    setQuotaLeft(null);
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_BASE}/stats`);
      setStats(res.data);
    } catch (err) {
      console.error("Error fetching stats:", err);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    // Insert a placeholder AI message we'll update as SSE events arrive.
    setMessages(prev => [...prev, { role: 'ai', content: '', sources: [], stage: 'Starting...' }]);

    const updateLastAI = (patch) => {
      setMessages(prev => {
        const next = [...prev];
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].role === 'ai') {
            next[i] = typeof patch === 'function' ? patch(next[i]) : { ...next[i], ...patch };
            break;
          }
        }
        return next;
      });
    };

    try {
      const history = messages.map(m => ({
        role: m.role === 'ai' ? 'assistant' : 'user',
        content: m.content,
      }));

      const token = await getAccessToken();
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ query: input, history }),
      });

      // 401 = anonymous trial used up; 429 = burst or daily quota. Both come
      // back as plain JSON rather than an SSE stream.
      if (res.status === 401 || res.status === 429) {
        const body = await res.json().catch(() => ({}));
        const msg = body.message || 'Rate limit reached.';
        updateLastAI({ content: msg, stage: null });
        if (res.status === 401) promptSignIn(msg);
        setQuotaLeft(0);
        return;
      }

      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`);
      }

      const remainingHeader = res.headers.get('X-Quota-Remaining');
      if (remainingHeader !== null) setQuotaLeft(Number(remainingHeader));

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      // Parse SSE stream: blocks separated by blank lines, each with `event:` + `data:` lines.
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sepIdx;
        while ((sepIdx = buffer.indexOf('\n\n')) !== -1) {
          const block = buffer.slice(0, sepIdx);
          buffer = buffer.slice(sepIdx + 2);
          let event = 'message';
          let data = '';
          for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) event = line.slice(7).trim();
            else if (line.startsWith('data: ')) data += line.slice(6);
          }
          if (!data) continue;
          let payload;
          try { payload = JSON.parse(data); } catch { continue; }

          if (event === 'stage') {
            updateLastAI({ stage: payload.label });
          } else if (event === 'token') {
            updateLastAI(prev => ({ ...prev, content: (prev.content || '') + (payload.text || '') }));
          } else if (event === 'replace') {
            updateLastAI({ content: '' });
          } else if (event === 'done') {
            updateLastAI({
              content: payload.answer,
              sources: payload.sources || [],
              stage: null,
              timings: payload.timings,
            });
          } else if (event === 'error') {
            updateLastAI({
              content: `Sorry, I encountered an error: ${payload.message || 'unknown'}`,
              stage: null,
            });
          }
        }
      }
    } catch (err) {
      updateLastAI({
        content: err.message || "Sorry, I encountered an error processing your request.",
        stage: null,
      });
    } finally {
      setLoading(false);
    }
  };

  const analyzeContent = async (text, action) => {
    try {
      setLoading(true);
      const token = await getAccessToken();
      const res = await axios.post(`${API_BASE}/analyze`, { text, action }, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      const aiMsg = {
        role: 'ai',
        content: `### ${action.toUpperCase()} ANALYSIS\n\n${res.data.result}`
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      if (err?.response?.status === 401) {
        promptSignIn(err.response.data?.message);
      }
      console.error("Analysis error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <button className="mobile-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
        {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
      </button>
      <div className={`sidebar-overlay ${sidebarOpen ? 'open' : ''}`} onClick={() => setSidebarOpen(false)} />
      <div className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="logo">
          JFK Files Research System
          <span className="logo-sub">Declassified Document Archive</span>
        </div>

        {authEnabled && (
          <div className="account-box">
            {session ? (
              <>
                <div className="account-email" title={session.user?.email}>
                  {session.user?.email}
                </div>
                <button className="account-btn" onClick={signOut}>
                  <LogOut size={13} /> Sign out
                </button>
              </>
            ) : (
              <>
                <div className="account-trial">
                  {quotaLeft === null
                    ? 'Unauthenticated access'
                    : quotaLeft > 0
                      ? `${quotaLeft} unauthenticated request${quotaLeft === 1 ? '' : 's'} left`
                      : 'Unauthenticated limit reached'}
                </div>
                <button className="account-btn" onClick={() => promptSignIn('')}>
                  <LogIn size={13} /> Sign in
                </button>
              </>
            )}
          </div>
        )}

        <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', borderLeft: '1px solid var(--border-light)', paddingLeft: '0.75rem' }}>
          <p style={{ fontWeight: '600', color: 'var(--text-muted)', marginBottom: '0.15rem' }}>Master of Statistics & Data Science</p>
          <p style={{ marginBottom: '0.35rem' }}>KU Leuven</p>
          <p style={{ fontStyle: 'italic', color: 'var(--text-dim)' }}>Thesis: "Topic Modeling and Thematic Analysis of JFK Assassination Files Using NLP"</p>
        </div>

        <div className="stats-section">
          <h3 style={{ marginBottom: '0.75rem', color: 'var(--text-dim)', fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.15em' }}>Archive Statistics</h3>
          <div style={{ display: 'grid', gap: '0.75rem' }}>
            <div className="stat-card">
              <span className="stat-value">{stats?.total_docs?.toLocaleString() || '---'}</span>
              <span className="stat-label">Documents</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">{stats?.total_pages?.toLocaleString() || '---'}</span>
              <span className="stat-label">Pages</span>
            </div>
          </div>
        </div>

        {/* Chat list */}
        <div className="chats-section">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <h3 style={{ color: 'var(--text-dim)', fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.15em' }}>Chats</h3>
            <button className="tool-btn" onClick={createNewChat} style={{ padding: '0.3rem 0.5rem' }}>
              <Plus size={12} /> New
            </button>
          </div>
          <div className="chat-list">
            {chats.map(chat => (
              <div
                key={chat.id}
                className={`chat-list-item ${chat.id === activeChatId ? 'active' : ''}`}
                onClick={() => { setActiveChatId(chat.id); setSidebarOpen(false); }}
              >
                <MessageSquare size={12} />
                <span className="chat-list-title">{chatTitle(chat.messages)}</span>
                <button
                  className="chat-delete-btn"
                  onClick={(e) => deleteChat(chat.id, e)}
                >
                  <Trash2 size={11} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="tools-section" style={{ marginTop: 'auto' }}>
          <h3 style={{ marginBottom: '0.75rem', color: 'var(--text-dim)', fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.15em' }}>Tools</h3>
          <div style={{ display: 'grid', gap: '0.5rem' }}>
            <button className="tool-btn" onClick={() => downloadChat(messages)} disabled={messages.length === 0}>
              <Download size={14} /> Export PDF
            </button>
            <button className="tool-btn" onClick={() => analyzeContent(messages[messages.length - 1]?.content, 'names')} disabled={messages.length === 0}>
              <Users size={14} /> Extract Names
            </button>
            <button className="tool-btn" onClick={() => analyzeContent(messages[messages.length - 1]?.content, 'summarize')} disabled={messages.length === 0}>
              <FileText size={14} /> Summarize
            </button>
          </div>
        </div>
      </div>

      <div className="main-content">
        <div className="chat-history">
          {messages.length === 0 && (
            <div className="welcome-screen">
              <div className="welcome-stamp">Declassified</div>
              <h2>JFK Files Research System</h2>
              <p>Query the declassified JFK assassination document archive. Ask about specific documents, individuals, events, or request analysis of classified materials.</p>
              <div className="sample-prompts">
                {[
                  "What was Oswald's connection to the Soviet embassy in Mexico City?",
                  "Show me document 104-10004-10143",
                  "How many pages include handwriting?",
                  "Why did Jack Ruby kill Oswald?",
                ].map((prompt, i) => (
                  <button
                    key={i}
                    className="sample-prompt-btn"
                    onClick={() => { setInput(prompt); }}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}
          <AnimatePresence>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                className={`message ${msg.role}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="msg-content">
                  {msg.role === 'ai' && msg.stage && (
                    <div className="stage-indicator" style={{ opacity: 0.7, fontSize: '0.85em', fontStyle: 'italic', marginBottom: msg.content ? '0.5rem' : 0 }}>
                      {msg.stage}
                    </div>
                  )}
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ node, children, href, title, ...props }) => (
                        <a href={href} title={title} target="_blank" rel="noopener noreferrer" {...props}>
                          {children}
                        </a>
                      )
                    }}
                  >
                    {msg.sources ? injectCitationLinks(msg.content, msg.sources) : msg.content}
                  </ReactMarkdown>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={chatEndRef} />
        </div>

        <div className="input-container">
          <form className="input-wrapper" onSubmit={handleSend}>
            <Search size={16} color="var(--text-dim)" />
            <input
              type="text"
              placeholder="Search declassified documents..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button type="submit" className="send-btn">
              <Send size={14} />
            </button>
          </form>
          <div className="copyright">
            © 2026 Furkan Demir · KU Leuven · All rights reserved · For academic research purposes only.
          </div>
        </div>
      </div>

      {authEnabled && (
        <Auth open={authOpen} onClose={() => setAuthOpen(false)} reason={authReason} />
      )}
    </div>
  );
}

export default App;
