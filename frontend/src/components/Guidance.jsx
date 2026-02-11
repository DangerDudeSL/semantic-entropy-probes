import React from 'react';
import { X, Info, AlertTriangle, ShieldCheck } from 'lucide-react';

const Guidance = ({ onClose }) => {
    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-3xl max-w-2xl w-full shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-[#f5f5f7]">
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                        <Info className="text-[#0071e3]" size={24} />
                        Understanding Uncertainty Metrics
                    </h2>
                    <button onClick={onClose} className="p-2 hover:bg-gray-200 rounded-full transition-colors">
                        <X size={20} />
                    </button>
                </div>

                <div className="p-6 space-y-6 overflow-y-auto max-h-[80vh]">

                    {/* Semantic Entropy Section */}
                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">1. Semantic Entropy (Confusion)</h3>
                        <div className="bg-amber-50 border border-amber-100 rounded-2xl p-4">
                            <p className="text-gray-700 leading-relaxed">
                                <strong>What it means:</strong> This measures how "confused" the model is about the <em>meaning</em> of its answer.
                            </p>
                            <p className="text-gray-600 mt-2 text-sm">
                                If the model generates 5 different sentences, do they all mean the same thing?
                                <br />
                                • <strong>Low Entropy (0.0):</strong> The model is consistent. All generations have the same meaning.
                                <br />
                                • <strong>High Entropy (&gt; 0.5):</strong> The model is hallucinating or confused. It is generating answers with conflicting meanings.
                            </p>
                        </div>
                    </section>

                    {/* Confidence/Accuracy Section */}
                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">2. Confidence (Accuracy Probability)</h3>
                        <div className="bg-blue-50 border border-blue-100 rounded-2xl p-4">
                            <p className="text-gray-700 leading-relaxed">
                                <strong>What it means:</strong> This is a score predicted by a probe trained to spot "correct" answers.
                            </p>
                            <p className="text-gray-600 mt-2 text-sm">
                                • <strong>High Confidence (&gt; 90%):</strong> The probe believes this sentence looks like a correct answer.
                                <br />
                                • <strong>Low Confidence (&lt; 50%):</strong> The probe spots patterns often found in incorrect answers.
                            </p>
                        </div>
                    </section>

                    {/* Contradiction Section */}
                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800 flex items-center gap-2">
                            <AlertTriangle size={20} className="text-red-500" />
                            What if they match?
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="p-4 border rounded-xl bg-green-50/50 border-green-100">
                                <div className="font-medium text-green-800 mb-1 flex items-center gap-2">
                                    <ShieldCheck size={16} /> Reliable
                                </div>
                                <div className="text-xs text-green-700">
                                    Low Entropy + High Confidence. The model is consistent and the answer looks correct.
                                </div>
                            </div>

                            <div className="p-4 border rounded-xl bg-red-50/50 border-red-100">
                                <div className="font-medium text-red-800 mb-1 flex items-center gap-2">
                                    <AlertTriangle size={16} /> Hallucination
                                </div>
                                <div className="text-xs text-red-700">
                                    High Entropy + Low Confidence. The model is guessing and confusing itself.
                                </div>
                            </div>
                        </div>
                    </section>

                </div>

                <div className="p-4 bg-gray-50 border-t flex justify-end">
                    <button
                        onClick={onClose}
                        className="px-6 py-2 bg-[#1d1d1f] text-white rounded-full hover:bg-black transition-all font-medium"
                    >
                        Got it
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Guidance;
