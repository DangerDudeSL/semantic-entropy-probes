
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import ChatInterface from './components/ChatInterface';
import UncertaintyChart from './components/UncertaintyChart';
import Guidance from './components/Guidance';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]); // Stores entropy metrics for chart
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ model_loaded: false });
  const [showGuidance, setShowGuidance] = useState(false);
  const [highlightEnabled, setHighlightEnabled] = useState(() => {
    const saved = localStorage.getItem('highlight_enabled');
    return saved !== null ? JSON.parse(saved) : true;
  });

  // Load history from local storage on mount
  useEffect(() => {
    const saved = localStorage.getItem('chat_history');
    if (saved) {
      const parsed = JSON.parse(saved);
      setMessages(parsed.messages || []);
      setHistory(parsed.history || []);
    }
    checkStatus();

    // Status Polling (Every 2 seconds)
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // Save to local storage whenever changed
  useEffect(() => {
    localStorage.setItem('chat_history', JSON.stringify({ messages, history }));
  }, [messages, history]);

  // Save highlight preference
  useEffect(() => {
    localStorage.setItem('highlight_enabled', JSON.stringify(highlightEnabled));
  }, [highlightEnabled]);

  const checkStatus = async () => {
    try {
      const res = await axios.get(`${API_URL}/status`);
      setStatus(res.data);
    } catch (e) {
      console.error("Backend offline");
      setStatus({ model_loaded: false });
    }
  };

  const handleSendMessage = async (text) => {
    // Add User Message
    const userMsg = { role: 'user', content: text, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/infer`, { question: text });

      const botMsg = {
        role: 'assistant',
        content: res.data.answer || "Error generating response",
        entropy: res.data.entropy,
        accuracy_prob: res.data.accuracy_prob,
        sentence_details: res.data.sentence_details,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMsg]);

      // Update Chart History
      setHistory(prev => [...prev, {
        question: text.substring(0, 15) + "...",
        entropy: res.data.entropy,
        accuracy: res.data.accuracy_prob
      }]);

    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not reach model.", isError: true }]);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    if (window.confirm("Clear all history?")) {
      setMessages([]);
      setHistory([]);
      localStorage.removeItem('chat_history');
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#f5f5f7] text-[#1d1d1f]">
      <Navbar
        status={status}
        onClear={clearHistory}
        onShowGuidance={() => setShowGuidance(true)}
        currentModel={status.model_name}
        highlightEnabled={highlightEnabled}
        onToggleHighlight={() => setHighlightEnabled(prev => !prev)}
      />

      {showGuidance && <Guidance onClose={() => setShowGuidance(false)} />}

      <main className="flex-1 flex max-w-[1600px] mx-auto w-full p-6 gap-6 h-[calc(100vh-80px)]">

        {/* Left: Chat Area */}
        <div className="flex-[2] flex flex-col glass rounded-3xl overflow-hidden shadow-sm h-full max-w-[65%]">
          <ChatInterface
            messages={messages}
            loading={loading}
            onSend={handleSendMessage}
            highlightEnabled={highlightEnabled}
          />
        </div>

        {/* Right: Metrics Panel */}
        <div className="flex-1 flex flex-col gap-6 h-full min-w-[350px]">
          <div className="glass rounded-3xl p-6 flex-1 shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Uncertainty Metrics</h2>
            <div className="h-full max-h-[400px]">
              <UncertaintyChart data={history} />
            </div>
          </div>

          {/* Current Stats Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-3xl p-6 shadow-sm"
          >
            <h3 className="text-gray-500 text-sm font-medium uppercase tracking-wider mb-2">Last Response</h3>
            {messages.length > 0 && messages[messages.length - 1].role === 'assistant' ? (
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-2xl bg-white/50">
                  <div className="text-sm text-gray-400">Semantic Entropy</div>
                  <div className={`text-2xl font-bold ${messages[messages.length - 1].entropy > 0.5 ? 'text-red-500' : 'text-green-500'}`}>
                    {messages[messages.length - 1].entropy?.toFixed(4) || "N/A"}
                  </div>
                </div>
                <div className="p-4 rounded-2xl bg-white/50">
                  <div className="text-sm text-gray-400">Accuracy Prob</div>
                  <div className="text-2xl font-bold text-blue-500">
                    {messages[messages.length - 1].accuracy_prob?.toFixed(4) || "N/A"}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-400 italic">No data yet...</div>
            )}
          </motion.div>
        </div>

      </main>
    </div>
  );
}

export default App;
