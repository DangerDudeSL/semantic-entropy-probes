
import React, { useEffect, useRef } from 'react';
import { Send, Bot, User, AlertCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

const ChatInterface = ({ messages, loading, onSend, highlightEnabled }) => {
    const [input, setInput] = React.useState('');
    const endRef = useRef(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!input.trim() || loading) return;
        onSend(input);
        setInput('');
    };

    return (
        <div className="flex flex-col h-full">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-center opacity-30 mt-[-50px]">
                        <Bot size={64} className="mb-4 text-[#0071e3]" />
                        <h2 className="text-2xl font-semibold">Semantic Probe AI</h2>
                        <p className="max-w-md mt-2">Ask a question to analyze model uncertainty in real-time.</p>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        key={idx}
                        className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                    >
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 
                        ${msg.role === 'user' ? 'bg-[#1d1d1f] text-white' : 'bg-[#0071e3] text-white'}`}>
                            {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                        </div>

                        <div className={`flex flex-col max-w-[80%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                            <div className={`rounded-2xl px-5 py-3 shadow-sm 
                            ${msg.role === 'user' ? 'bg-[#1d1d1f] text-white rounded-tr-none' : 'bg-white rounded-tl-none border border-gray-100'}`}>

                                {/* Content Rendering: Highlighted Sentences or Plain Text */}
                                {highlightEnabled && msg.sentence_details && msg.sentence_details.length > 0 ? (
                                    <div className="leading-relaxed">
                                        {msg.sentence_details.map((sent, sIdx) => {
                                            // Calculate opacity based on entropy (0.0 to 1.0)
                                            // Only highlight if entropy > 0.4 reasonable threshold
                                            let bgStyle = {};
                                            if (sent.entropy > 0.4) {
                                                // Red highlight scaling with entropy
                                                const opacity = Math.min((sent.entropy - 0.2), 0.5);
                                                bgStyle = { backgroundColor: `rgba(255, 59, 48, ${opacity})` };
                                            }
                                            return (
                                                <span
                                                    key={sIdx}
                                                    style={bgStyle}
                                                    className="transition-colors duration-300 rounded px-0.5 cursor-help border-b border-transparent hover:border-red-400"
                                                    title={`Entropy: ${sent.entropy.toFixed(3)} | Confidence: ${(sent.accuracy_prob * 100).toFixed(1)}%`}
                                                >
                                                    {sent.text}{" "}
                                                </span>
                                            );
                                        })}
                                    </div>
                                ) : (
                                    msg.content
                                )}
                            </div>

                            {/* Metrics Badge for Assistant */}
                            {msg.role === 'assistant' && !msg.isError && (
                                <div className="mt-2 text-sm">
                                    {(() => {
                                        const ent = msg.entropy;
                                        const acc = msg.accuracy_prob;
                                        let label = "Reliable";
                                        let colorClass = "text-green-600 bg-green-50 border-green-200";
                                        let icon = "🛡️";

                                        if (ent > 0.45) {
                                            label = "Uncertain / Confused";
                                            colorClass = "text-amber-600 bg-amber-50 border-amber-200";
                                            icon = "⚠️";
                                        } else if (acc < 0.6) {
                                            label = "Likely Incorrect/Hallucination";
                                            colorClass = "text-red-600 bg-red-50 border-red-200";
                                            icon = "🚨";
                                        }

                                        return (
                                            <div className="flex items-center gap-3 select-none">
                                                <div className={`px-3 py-1.5 rounded-full border flex items-center gap-2 font-medium ${colorClass}`}>
                                                    <span>{icon}</span>
                                                    <span>{label}</span>
                                                </div>
                                                <div className="text-xs text-gray-400 flex gap-2">
                                                    <span>Entropy: {ent.toFixed(2)}</span>
                                                    <span>Conf: {(acc * 100).toFixed(0)}%</span>
                                                </div>
                                            </div>
                                        );
                                    })()}
                                </div>
                            )}
                            {msg.isError && (
                                <div className="mt-1 text-xs text-red-500 flex items-center gap-1">
                                    <AlertCircle size={12} /> Failed to inference
                                </div>
                            )}
                        </div>
                    </motion.div>
                ))}

                {loading && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
                        <div className="w-8 h-8 rounded-full bg-[#0071e3] text-white flex items-center justify-center shrink-0">
                            <Bot size={16} />
                        </div>
                        <div className="bg-white rounded-2xl rounded-tl-none px-5 py-4 shadow-sm border border-gray-100 flex items-center gap-2">
                            <Loader2 className="animate-spin text-gray-400" size={18} />
                            <span className="text-sm text-gray-400">Analyzing semantics...</span>
                        </div>
                    </motion.div>
                )}
                <div ref={endRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-white/50 border-t border-gray-100 backdrop-blur-sm">
                <form onSubmit={handleSubmit} className="relative flex items-center">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask something..."
                        disabled={loading}
                        className="w-full bg-white border border-gray-200 rounded-full pl-6 pr-14 py-4 focus:outline-none focus:ring-2 focus:ring-[#0071e3]/20 focus:border-[#0071e3] transition-all shadow-sm"
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || loading}
                        className="absolute right-2 p-2 bg-[#0071e3] text-white rounded-full hover:bg-[#0077ED] disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                    >
                        <Send size={20} />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatInterface;
