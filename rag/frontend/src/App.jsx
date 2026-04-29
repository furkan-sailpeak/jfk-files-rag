import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Users, FileText, Search, Menu, X, Plus, Download, MessageSquare, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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

function downloadChat(messages) {
  let md = '# JFK Files Research — Chat Export\n\n';
  md += `_Exported: ${new Date().toLocaleString()}_\n\n---\n\n`;

  for (const msg of messages) {
    if (msg.role === 'user') {
      md += `## Q: ${msg.content}\n\n`;
    } else {
      md += `${msg.content}\n\n`;
      if (msg.sources && msg.sources.length > 0) {
        md += '**Sources:**\n';
        msg.sources.forEach((s, i) => {
          md += `- [${i + 1}] ${s.filename}, Page ${s.page}\n`;
        });
        md += '\n';
      }
      md += '---\n\n';
    }
  }

  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `jfk-research-${new Date().toISOString().slice(0, 10)}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

function App() {
  const [chats, setChats] = useState([{ id: generateId(), messages: [] }]);
  const [activeChatId, setActiveChatId] = useState(chats[0].id);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
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

      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input, history }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`);
      }

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
      const res = await axios.post(`${API_BASE}/analyze`, { text, action });
      const aiMsg = {
        role: 'ai',
        content: `### ${action.toUpperCase()} ANALYSIS\n\n${res.data.result}`
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
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
              <Download size={14} /> Export Chat
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
    </div>
  );
}

export default App;
