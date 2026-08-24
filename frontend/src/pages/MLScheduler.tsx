import React from 'react';
import type { SchedulerDecision, DecisionTimelineEntry } from '../types/scheduler';
import type { RewardDataPoint } from '../types/performance';
import { 
  Cpu, 
  Target, 
  Compass, 
  Award, 
  Clock, 
  HelpCircle
} from 'lucide-react';

import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  Legend 
} from 'recharts';

interface MLSchedulerProps {
  decision: SchedulerDecision;
  timeline: DecisionTimelineEntry[];
  rewardTrajectory: RewardDataPoint[];
}

export const MLScheduler: React.FC<MLSchedulerProps> = ({
  decision,
  timeline,
  rewardTrajectory,
}) => {
  const isExploitation = decision.strategy === 'EXPLOITATION';
  const explorationPct = Math.round(decision.exploration_probability * 100);
  const exploitationPct = 100 - explorationPct;

  // Chart data for cumulative reward
  const chartData = rewardTrajectory.slice(-30);

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-charcoal-750/80">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-rf-green-bg border border-rf-green-border text-rf-green">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h2 className="font-display font-bold text-lg text-slate-100 uppercase tracking-wide">
              ADAPTIVE SCAN SCHEDULER & REINFORCEMENT ENGINE
            </h2>
            <p className="text-xs text-slate-400 font-sans">
              Exploitation vs Exploration Policy & Reward Optimization
            </p>
          </div>
        </div>

        <span className="px-3 py-1 rounded-md bg-rf-green-bg text-rf-green border border-rf-green-border text-xs font-mono font-bold">
          CLOSED-LOOP SCHEDULER ACTIVE
        </span>
      </div>

      {/* Section A & B: Current Decision + Exploration/Exploitation Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* A. CURRENT DECISION */}
        <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-charcoal-750/80">
            <div className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <Target className="w-4 h-4 text-rf-green" />
              <span>A. CURRENT ML DECISION STATE</span>
            </div>
            <span
              className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold ${
                isExploitation
                  ? 'bg-rf-green-bg text-rf-green-light border border-rf-green-border'
                  : 'bg-rf-amber-bg text-rf-amber-light border border-rf-amber-border'
              }`}
            >
              {decision.strategy}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-charcoal-850 border border-charcoal-700">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Selected Band</div>
              <div className="text-xl font-mono font-black text-rf-green-light mt-1">
                BAND {decision.next_band}
              </div>
            </div>

            <div className="p-3 rounded-lg bg-charcoal-850 border border-charcoal-700">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Action</div>
              <div className="text-xl font-mono font-black text-slate-200 mt-1">
                SCAN
              </div>
            </div>

            <div className="p-3 rounded-lg bg-charcoal-850 border border-charcoal-700">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Est. Hit Probability</div>
              <div className="text-xl font-mono font-black text-rf-cyan-light mt-1">
                {(decision.estimated_hit_probability * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          <div className="p-3.5 rounded-lg bg-charcoal-850/60 border border-charcoal-750 text-xs">
            <div className="text-[10px] font-mono uppercase text-slate-400 font-semibold mb-1 flex items-center gap-1">
              <HelpCircle className="w-3.5 h-3.5 text-rf-cyan" />
              <span>Decision Explanation</span>
            </div>
            <p className="text-slate-300 font-sans leading-relaxed italic">
              "{decision.reasoning}"
            </p>
          </div>
        </div>

        {/* B. EXPLORATION VS EXPLOITATION */}
        <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-charcoal-750/80">
            <div className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <Compass className="w-4 h-4 text-rf-amber" />
              <span>B. EXPLORATION VS EXPLOITATION (ε-GREEDY)</span>
            </div>
            <span className="text-xs font-mono text-slate-400">ε = {(decision.exploration_probability).toFixed(2)}</span>
          </div>

          {/* Visual Ratio Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono font-bold">
              <span className="text-rf-green-light">Exploitation: {exploitationPct}%</span>
              <span className="text-rf-amber-light">Exploration: {explorationPct}%</span>
            </div>

            <div className="h-4 w-full bg-charcoal-950 rounded-full overflow-hidden p-0.5 border border-charcoal-750 flex">
              <div
                style={{ width: `${exploitationPct}%` }}
                className="h-full bg-gradient-to-r from-rf-green-dark via-rf-green to-rf-green-light rounded-l-full"
              />
              <div
                style={{ width: `${explorationPct}%` }}
                className="h-full bg-gradient-to-r from-rf-amber via-rf-amber-light to-yellow-300 rounded-r-full"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-1 text-xs">
            <div className="p-3 rounded-lg bg-charcoal-850/80 border border-charcoal-750">
              <div className="font-mono font-bold text-rf-green-light mb-1">Exploitation (1 - ε)</div>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Prioritizes historically high-yield emitter bands (e.g. Band 7 & 3) to harvest maximum reward and intercept throughput.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-charcoal-850/80 border border-charcoal-750">
              <div className="font-mono font-bold text-rf-amber-light mb-1">Exploration (ε)</div>
              <p className="text-slate-400 text-[11px] leading-relaxed">
                Periodically tests uncertain, high-staleness frequency bands to discover newly activated or frequency-agile radar emitters.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section C & D: Learning Curves & Reward Optimization */}
      <div className="p-6 rounded-xl bg-charcoal-900 border border-charcoal-750/80 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-charcoal-750/80">
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-rf-green" />
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              C & D. CUMULATIVE REWARD & LEARNING CURVE
            </h3>
          </div>
          <span className="text-xs font-mono text-slate-400">Step vs Cumulative Intercept Reward</span>
        </div>

        <div className="h-64 w-full bg-charcoal-950/90 rounded-lg p-3 border border-charcoal-800">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 10, fill: '#94a3b8' }} label={{ value: 'Simulation Step', position: 'insideBottom', offset: -4, fill: '#64748b', fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0B0F14', borderColor: '#233041', borderRadius: '8px', fontSize: '11px', fontFamily: 'monospace' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace' }} />
              <Line type="monotone" dataKey="ml_cumulative" name="Smart ML Scheduler (Cumulative J)" stroke="#10b981" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="baseline_cumulative" name="Traditional Baseline (Cumulative J)" stroke="#64748b" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Section E: Decision Timeline */}
      <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80">
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-charcoal-750/80">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-rf-cyan" />
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              E. DECISION TIMELINE AUDIT FEED
            </h3>
          </div>
          <span className="text-xs font-mono text-slate-400">Step-by-Step Decision History</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2.5">
          {timeline.slice(-16).reverse().map((entry) => (
            <div
              key={entry.step}
              className={`p-2.5 rounded-lg border text-xs font-mono space-y-1 ${
                entry.result === 'HIT'
                  ? 'bg-rf-green-bg border-rf-green-border text-rf-green-light'
                  : 'bg-charcoal-850 border-charcoal-750 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between text-[10px] text-slate-400">
                <span>Step #{entry.step}</span>
                <span className="font-bold">{entry.result}</span>
              </div>
              <div className="text-sm font-bold text-slate-100">
                BAND {entry.selected_band}
              </div>
              <div className="text-[10px] text-slate-400">
                Prob: {(entry.hit_probability * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
