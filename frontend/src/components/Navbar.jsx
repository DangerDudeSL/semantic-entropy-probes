
import React, { useState } from 'react';
import { Bot, HelpCircle, ChevronDown, Check, Eye, EyeOff } from 'lucide-react';
import axios from 'axios';

const API_URL = "http://localhost:8000";

const Navbar = ({ status, onClear, onShowGuidance, currentModel, highlightEnabled, onToggleHighlight }) => {
    const [switching, setSwitching] = useState(false);
    const [msg, setMsg] = useState("");

    const handleModelChange = async (e) => {
        const newModel = e.target.value;
        if (newModel === currentModel) return;

        if (!window.confirm(`Switch to ${newModel}? This will reload the model and takes time.`)) {
            // Reset select
            e.target.value = currentModel;
            return;
        }

        setSwitching(true);
        setMsg("Switching Model...");
        try {
            await axios.post(`${API_URL}/set_model`, { model_name: newModel });
            setMsg("Model Switched!");
            setTimeout(() => setMsg(""), 3000);
            // Reload page or let poller update status
            window.location.reload();
        } catch (e) {
            setMsg("Failed to switch.");
            setSwitching(false);
        }
    };

    return (
        <nav className="h-[80px] flex items-center justify-between px-8 bg-white/80 backdrop-blur-md border-b border-gray-100 sticky top-0 z-40">

            <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[#1d1d1f] rounded-xl flex items-center justify-center text-white shadow-lg shadow-black/10">
                    <Bot size={24} />
                </div>
                <div>
                    <h1 className="font-semibold text-lg tracking-tight text-[#1d1d1f]">Semantic Probes</h1>
                    <div className="flex items-center gap-2 text-xs">
                        <span className={`w-2 h-2 rounded-full ${status.model_loaded ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-red-500'}`}></span>
                        {switching ? (
                            <span className="text-amber-600 font-medium animate-pulse">{msg}...</span>
                        ) : (
                            <select
                                value={currentModel}
                                onChange={handleModelChange}
                                className="bg-transparent text-gray-500 font-medium focus:outline-none cursor-pointer hover:text-gray-700 transition-colors appearance-none pr-4"
                                style={{ backgroundImage: 'none' }} // Hide default arrow if desired, or keep it
                            >
                                <option value="meta-llama/Llama-2-7b-hf">Llama 2 (7B)</option>
                                <option value="meta-llama/Meta-Llama-3.1-8B-Instruct">Llama 3.1 (8B)</option>
                            </select>
                        )}
                        {!switching && <ChevronDown size={12} className="text-gray-400 -ml-3 pointer-events-none" />}
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-4">
                {/* Highlight Toggle */}
                <button
                    onClick={onToggleHighlight}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all duration-300 border ${highlightEnabled
                            ? 'bg-blue-50 text-blue-600 border-blue-200 hover:bg-blue-100'
                            : 'bg-gray-50 text-gray-400 border-gray-200 hover:bg-gray-100'
                        }`}
                    title={highlightEnabled ? 'Sentence highlighting is ON' : 'Sentence highlighting is OFF'}
                >
                    {highlightEnabled ? <Eye size={14} /> : <EyeOff size={14} />}
                    <span className="hidden sm:inline">{highlightEnabled ? 'Highlights On' : 'Highlights Off'}</span>
                </button>

                <button
                    onClick={onShowGuidance}
                    className="p-2 text-gray-400 hover:text-[#0071e3] transition-colors rounded-full hover:bg-gray-100"
                    title="Help & Guidance"
                >
                    <HelpCircle size={22} />
                </button>

                <button
                    onClick={onClear}
                    className="px-4 py-2 text-sm font-medium text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                >
                    Clear History
                </button>
            </div>
        </nav>
    );
};

export default Navbar;
