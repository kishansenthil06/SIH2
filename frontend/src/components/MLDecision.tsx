import React from 'react';
import type { SchedulerDecision } from '../types/scheduler';
import { Cpu, Compass, Target, HelpCircle } from 'lucide-react';


interface MLDecisionProps {
  decision: SchedulerDecision;
}

export const MLDecision: React.FC<MLDecisionProps> = ({ decision }) => {
  const isExploitation = decision.strategy === 'EXPLOITATION';

  return (
    <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-charcoal-750/80">
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 rounded-md bg-rf-green-bg border border-rf-green-border text-rf-green">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              ML SCHEDULER DECISION
            </h2>
            <p className="text-[11px] text-slate-400 font-sans">Reinforcement Next-Band Selection</p>
          </div>
        </div>

        {/* Strategy Badge */}
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-semibold ${
            isExploitation
              ? 'bg-rf-green-bg text-rf-green-light border border-rf-green-border'
              : 'bg-rf-amber-bg text-rf-amber-light border border-rf-amber-border'
          }`}
        >
          {isExploitation ? <Target className="w-3.5 h-3.5" /> : <Compass className="w-3.5 h-3.5" />}
          {decision.strategy}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 items-center">
        {/* Left Side: Decision Card */}
        <div className="space-y-3">
          <div className="p-4 rounded-lg bg-charcoal-850 border border-charcoal-700 flex items-center justify-between">
            <div>
              <div className="text-[10px] font-mono uppercase text-slate-400 font-semibold">Recommended Next Scan</div>
              <div className="text-2xl font-mono font-black text-rf-green-light mt-0.5">
                BAND {decision.next_band}
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] font-mono uppercase text-slate-400 font-semibold">Estimated Hit Probability</div>
              <div className="text-2xl font-mono font-black text-slate-100 mt-0.5">
                {(decision.estimated_hit_probability * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Reasoning Text Box */}
          <div className="p-3 rounded-lg bg-charcoal-850/60 border border-charcoal-750 text-xs">
            <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase text-slate-400 font-semibold mb-1">
              <HelpCircle className="w-3 h-3 text-rf-cyan" />
              <span>Scheduler Rationale</span>
            </div>
            <p className="text-slate-300 font-sans text-[11px] leading-relaxed italic">
              "{decision.reasoning}"
            </p>
          </div>

          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 px-1">
            <span>Exploration Rate (ε): <strong className="text-slate-200">{(decision.exploration_probability * 100).toFixed(0)}%</strong></span>
            <span>Exploitation Confidence: <strong className="text-rf-green-light">{((1 - decision.exploration_probability) * 100).toFixed(0)}%</strong></span>
          </div>
        </div>

        {/* Right Side: Recent Top Band Probability Ranking Bars */}
        <div className="p-4 rounded-lg bg-charcoal-850/80 border border-charcoal-750">
          <div className="text-[10px] font-mono uppercase text-slate-400 font-semibold mb-3 flex items-center justify-between">
            <span>Recent Band Expected Value</span>
            <span>Hit Prob</span>
          </div>

          <div className="space-y-2.5">
            {decision.band_probabilities.map((item) => {
              const isTop = item.band === decision.next_band;
              const probPct = Math.round(item.probability * 100);
              return (
                <div key={item.band} className="space-y-1">
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className={`font-semibold ${isTop ? 'text-rf-green-light' : 'text-slate-300'}`}>
                      Band {item.band} {isTop && <span className="text-[9px] text-rf-green font-normal">(Selected)</span>}
                    </span>
                    <span className="font-bold text-slate-200">{probPct}%</span>
                  </div>
                  <div className="w-full h-2 bg-charcoal-950 rounded-full overflow-hidden p-0.5 border border-charcoal-700">
                    <div
                      style={{ width: `${probPct}%` }}
                      className={`h-full rounded-full transition-all duration-300 ${
                        isTop
                          ? 'bg-gradient-to-r from-rf-green via-rf-green-light to-rf-cyan shadow-glow-green'
                          : 'bg-slate-500'
                      }`}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
