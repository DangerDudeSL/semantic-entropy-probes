import React from 'react';
import { X, Info, AlertTriangle, ShieldCheck } from 'lucide-react';

const Guidance = ({ onClose }) => {
    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-3xl max-w-2xl w-full shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-[#f5f5f7]">
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                        <Info className="text-[#0071e3]" size={24} />
                        Understanding Confidence Scores
                    </h2>
                    <button onClick={onClose} className="p-2 hover:bg-gray-200 rounded-full transition-colors">
                        <X size={20} />
                    </button>
                </div>

                <div className="p-6 space-y-6 overflow-y-auto max-h-[80vh]">

                    {/* What is Confidence */}
                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">How Confidence Works</h3>
                        <div className="bg-blue-50 border border-blue-100 rounded-2xl p-4">
                            <p className="text-gray-700 leading-relaxed">
                                <strong>Confidence</strong> is a unified score that combines two probe signals:
                                <strong> semantic entropy</strong> (how confused the model is) and <strong>accuracy probability</strong> (how correct the answer looks).
                            </p>
                            <p className="text-gray-600 mt-2 text-sm">
                                The aggregate score is the <strong>average confidence of all factual claim sentences</strong>,
                                ignoring filler, jokes, and meta-commentary. This gives a more reliable assessment than scoring the entire sequence at once.
                            </p>
                        </div>
                    </section>

                    {/* Score Ranges */}
                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">Score Ranges</h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                            <div className="p-4 border rounded-xl bg-green-50/50 border-green-100 text-center">
                                <div className="font-medium text-green-800 mb-1 flex items-center justify-center gap-2">
                                    <ShieldCheck size={16} /> Reliable
                                </div>
                                <div className="text-2xl font-bold text-green-600 my-1">&ge; 75%</div>
                                <div className="text-xs text-green-700">
                                    Model is consistent and confident in its answer.
                                </div>
                            </div>

                            <div className="p-4 border rounded-xl bg-amber-50/50 border-amber-100 text-center">
                                <div className="font-medium text-amber-800 mb-1 flex items-center justify-center gap-2">
                                    <AlertTriangle size={16} /> Uncertain
                                </div>
                                <div className="text-2xl font-bold text-amber-600 my-1">50–74%</div>
                                <div className="text-xs text-amber-700">
                                    Model shows some uncertainty. Verify with other sources.
                                </div>
                            </div>

                            <div className="p-4 border rounded-xl bg-red-50/50 border-red-100 text-center">
                                <div className="font-medium text-red-800 mb-1 flex items-center justify-center gap-2">
                                    <AlertTriangle size={16} /> Hallucinated
                                </div>
                                <div className="text-2xl font-bold text-red-600 my-1">&lt; 50%</div>
                                <div className="text-xs text-red-700">
                                    Model is likely guessing. Do not trust this answer.
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* SLT Score */}
                    <section>
                        <h3 className="text-lg font-semibold mb-2 text-gray-800">SLT Probe Score</h3>
                        <div className="bg-gray-50 border border-gray-200 rounded-2xl p-4">
                            <p className="text-gray-600 text-sm leading-relaxed">
                                The <strong>SLT Probe Score</strong> shown in the metrics panel is the confidence computed from the
                                second-to-last token of the entire generated sequence. This is the raw probe output and may be less reliable
                                for long, multi-sentence answers. It is provided as a reference for comparison.
                            </p>
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
