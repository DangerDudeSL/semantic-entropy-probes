
import React, { useEffect, useRef } from 'react';
import { Send, Bot, User, AlertCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

/**
 * Lightweight inline markdown renderer for LLM output.
 * Handles: **bold**, *italic*, `code`, and strips heading markers (###, ##, #).
 */
const renderInlineMarkdown = (text) => {
    if (!text) return text;

    // Process bold (**text**), italic (*text*), and code (`text`)
    const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
    let lastIndex = 0;
    let match;
    const parts = [];
    let key = 0;

    while ((match = regex.exec(text)) !== null) {
        if (match.index > lastIndex) {
            parts.push(text.slice(lastIndex, match.index));
        }
        if (match[2]) {
            parts.push(<strong key={key++}>{match[2]}</strong>);
        } else if (match[3]) {
            parts.push(<em key={key++}>{match[3]}</em>);
        } else if (match[4]) {
            parts.push(
                <code key={key++} className="bg-gray-100 text-red-600 px-1 py-0.5 rounded text-sm font-mono">
                    {match[4]}
                </code>
            );
        }
        lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex));
    }

    return parts.length > 0 ? parts : text;
};

/**
 * Detect what kind of "block" a sentence is and return rendering info.
 * Returns: { type, content, level }
 */
const classifySentence = (text) => {
    if (!text) return { type: 'text', content: text };

    // Headings: ### Heading, ## Heading, # Heading
    const headingMatch = text.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
        return {
            type: 'heading',
            content: headingMatch[2],
            level: headingMatch[1].length,
        };
    }

    // Bullet items: - item, * item, • item
    if (/^\s*[-•▪▸►]\s+/.test(text)) {
        return { type: 'bullet', content: text.replace(/^\s*[-•▪▸►]\s+/, '') };
    }

    // Numbered items: 1. item, 2) item, a) item
    const numberedMatch = text.match(/^\s*(\d{1,3}[.)]\s+|[a-zA-Z][.)]\s+)(.+)$/);
    if (numberedMatch) {
        return { type: 'numbered', label: numberedMatch[1].trim(), content: numberedMatch[2] };
    }

    // Star bullet (*) — only when at start and NOT bold pattern
    if (/^\s*\*\s+[^*]/.test(text)) {
        return { type: 'bullet', content: text.replace(/^\s*\*\s+/, '') };
    }

    return { type: 'text', content: text };
};

/**
 * Render a single sentence with proper formatting.
 * Applies heading/bullet styling + inline markdown.
 */
const renderFormattedSentence = (text) => {
    const info = classifySentence(text);
    const rendered = renderInlineMarkdown(info.content);

    switch (info.type) {
        case 'heading': {
            const sizes = {
                1: 'text-lg font-bold',
                2: 'text-base font-bold',
                3: 'text-sm font-bold uppercase tracking-wide text-gray-600',
                4: 'text-sm font-semibold text-gray-500',
            };
            return (
                <span className={`block mt-2 mb-1 ${sizes[info.level] || sizes[3]}`}>
                    {rendered}
                </span>
            );
        }
        case 'bullet':
            return <><span className="text-gray-400 mr-1">•</span>{rendered}</>;
        case 'numbered':
            return <><span className="text-gray-400 font-medium mr-1">{info.label}</span>{rendered}</>;
        default:
            return rendered;
    }
};

/** Check if a sentence needs its own line (headings, bullets, numbered items) */
const needsNewLine = (text) => {
    if (!text) return false;
    return /^(#{1,4}\s|[-•▪▸►*]\s|\s*\d{1,3}[.)]\s|[a-zA-Z][.)]\s)/.test(text.trim());
};

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
                                            const newLine = needsNewLine(sent.text);
                                            const renderedText = renderFormattedSentence(sent.text);

                                            // Non-claim sentences: dimmed, no highlight, no tooltip
                                            if (sent.is_claim === false) {
                                                return (
                                                    <React.Fragment key={sIdx}>
                                                        {newLine && <br />}
                                                        <span
                                                            className="text-gray-400 italic transition-colors duration-300 rounded px-0.5"
                                                            title="Non-claim (not scored)"
                                                        >
                                                            {renderedText}{" "}
                                                        </span>
                                                    </React.Fragment>
                                                );
                                            }

                                            // Claim sentences: highlight by confidence + hover tooltip
                                            let bgStyle = {};
                                            const conf = sent.confidence ?? 1;
                                            if (conf < 0.6) {
                                                const opacity = Math.min((1 - conf) * 0.8, 0.5);
                                                bgStyle = { backgroundColor: `rgba(255, 59, 48, ${opacity})` };
                                            }
                                            return (
                                                <React.Fragment key={sIdx}>
                                                    {newLine && <br />}
                                                    <span
                                                        style={bgStyle}
                                                        className="transition-colors duration-300 rounded px-0.5 cursor-help border-b border-transparent hover:border-red-400"
                                                        title={`Confidence: ${(conf * 100).toFixed(1)}% (Entropy: ${sent.entropy.toFixed(3)}, Accuracy: ${(sent.accuracy_prob * 100).toFixed(1)}%)`}
                                                    >
                                                        {renderedText}{" "}
                                                    </span>
                                                </React.Fragment>
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
                                        const conf = msg.confidence ?? 1;
                                        let label = "Reliable";
                                        let colorClass = "text-green-600 bg-green-50 border-green-200";
                                        let icon = "🛡️";

                                        if (conf < 0.5) {
                                            label = "Likely Hallucinated";
                                            colorClass = "text-red-600 bg-red-50 border-red-200";
                                            icon = "🚨";
                                        } else if (conf < 0.75) {
                                            label = "Uncertain";
                                            colorClass = "text-amber-600 bg-amber-50 border-amber-200";
                                            icon = "⚠️";
                                        }

                                        return (
                                            <div className="flex items-center gap-3 select-none">
                                                <div className={`px-3 py-1.5 rounded-full border flex items-center gap-2 font-medium ${colorClass}`}>
                                                    <span>{icon}</span>
                                                    <span>{label}</span>
                                                </div>
                                                <div className="text-xs text-gray-400">
                                                    <span>Confidence: {(conf * 100).toFixed(1)}%</span>
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
