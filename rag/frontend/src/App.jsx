import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import { Send, Users, FileText, Search, Menu, X } from 'lucide-react';
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

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const chatEndRef = useRef(null);

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

    try {
      const history = messages.map(m => ({
        role: m.role === 'ai' ? 'assistant' : 'user',
        content: m.content
      }));
      const res = await axios.post(`${API_BASE}/chat`, { query: input, history });
      const aiMsg = {
        role: 'ai',
        content: res.data.answer,
        sources: res.data.sources
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      const errorMsg = err.response?.data?.error || "Sorry, I encountered an error processing your request.";
      setMessages(prev => [...prev, { role: 'ai', content: errorMsg }]);
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

        <div className="tools-section" style={{ marginTop: 'auto' }}>
          <h3 style={{ marginBottom: '0.75rem', color: 'var(--text-dim)', fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.15em' }}>Analysis Tools</h3>
          <div style={{ display: 'grid', gap: '0.5rem' }}>
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
                  "Which documents contain redacted information?",
                  "Analyze the CIA's surveillance activities related to the assassination",
                  "Show me document 104-10004-10143",
                  "How many pages include handwriting?",
                  "What role did Jack Ruby play according to the files?",
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
          {loading && (
            <div className="message ai" style={{ opacity: 0.6 }}>
              <div className="typing-indicator">Searching classified archives...</div>
            </div>
          )}
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
        </div>
      </div>
    </div>
  );
}

export default App;
