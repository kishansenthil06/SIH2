import React from 'react';
import type { SimulationConfig, EventLogItem } from '../types/simulation';
import type { CurrentScan } from '../types/rf';
import { EventLog } from '../components/EventLog';
import { 
  PlaySquare, 
  Play, 
  Pause, 
  RotateCcw, 
  StepForward
} from 'lucide-react';


interface SimulationProps {
  isRunning: boolean;
  config: SimulationConfig;
  currentScan: CurrentScan;
  eventLogs: EventLogItem[];
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onStep: () => void;
  onUpdateStrategy: (strategy: SimulationConfig['strategy']) => void;
  onUpdateSpeed: (speed: number) => void;
  onUpdateEpsilon: (epsilon: number) => void;
}

export const Simulation: React.FC<SimulationProps> = ({
  isRunning,
  config,
  currentScan,
  eventLogs,
  onStart,
  onPause,
  onReset,
  onStep,
  onUpdateStrategy,
  onUpdateSpeed,
  onUpdateEpsilon,
}) => {
  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="pb-4 border-b border-charcoal-750/80">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-rf-green-bg border border-rf-green-border text-rf-green">
              <PlaySquare className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-display font-bold text-lg text-slate-100 uppercase tracking-wide">
                RF SIMULATION & RUNTIME CONTROL
              </h2>
              <p className="text-xs text-slate-400 font-sans">
                Interactive Parameter Tuning, Speed Control & Live Pipeline Telemetry
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {isRunning ? (
              <button
                onClick={onPause}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-rf-amber text-charcoal-950 font-mono font-bold text-xs shadow-glow-amber transition-all"
              >
                <Pause className="w-4 h-4" />
                <span>PAUSE RUN</span>
              </button>
            ) : (
              <button
                onClick={onStart}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-rf-green text-charcoal-950 font-mono font-bold text-xs shadow-glow-green transition-all"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>START SIMULATION</span>
              </button>
            )}

            <button
              onClick={onStep}
              disabled={isRunning}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-charcoal-800 hover:bg-charcoal-750 text-slate-300 border border-charcoal-700 disabled:opacity-40 text-xs font-mono font-semibold transition-colors"
            >
              <StepForward className="w-4 h-4" />
              <span>STEP</span>
            </button>

            <button
              onClick={onReset}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-charcoal-800 hover:bg-charcoal-750 text-slate-300 border border-charcoal-700 text-xs font-mono font-semibold transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              <span>RESET</span>
            </button>
          </div>
        </div>
      </div>

      {/* Control Panels Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Simulation Parameters & Controls */}
        <div className="lg:col-span-2 space-y-5">
          {/* Strategy Selection */}
          <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80 space-y-3">
            <div className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              1. Scan Strategy Selector
            </div>

            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => onUpdateStrategy('smart_ml')}
                className={`p-4 rounded-xl border text-left transition-all ${
                  config.strategy === 'smart_ml'
                    ? 'bg-charcoal-850 border-rf-green-border shadow-glow-green'
                    : 'bg-charcoal-850/60 border-charcoal-750 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-mono font-bold text-sm text-rf-green-light">
                    Smart ML Scheduler
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rf-green-bg text-rf-green border border-rf-green-border">
                    RECOMMENDED
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Closed-loop adaptive scheduler learning empirical hit probabilities with reinforcement feedback.
                </p>
              </button>

              <button
                onClick={() => onUpdateStrategy('sequential')}
                className={`p-4 rounded-xl border text-left transition-all ${
                  config.strategy === 'sequential'
                    ? 'bg-charcoal-850 border-slate-400 shadow-md'
                    : 'bg-charcoal-850/60 border-charcoal-750 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-mono font-bold text-sm text-slate-300">
                    Traditional Sequential Sweep
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-charcoal-750 text-slate-400">
                    BASELINE
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Fixed round-robin scanning visiting Band 1 → 20 in rigid sequential order without feedback.
                </p>
              </button>
            </div>
          </div>

          {/* Speed & Exploration Controls */}
          <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80 space-y-4">
            <div className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              2. Execution Speed & Exploration Parameters
            </div>

            {/* Speed Selector */}
            <div>
              <div className="text-[11px] font-mono text-slate-400 mb-2">Simulation Speed</div>
              <div className="grid grid-cols-4 gap-2 font-mono text-xs">
                {[
                  { speed: 1, label: 'Slow (1x)' },
                  { speed: 2, label: 'Normal (2x)' },
                  { speed: 5, label: 'Fast (5x)' },
                  { speed: 10, label: 'Max (10x)' },
                ].map((s) => (
                  <button
                    key={s.speed}
                    onClick={() => onUpdateSpeed(s.speed)}
                    className={`py-2 rounded-lg border font-semibold transition-all ${
                      config.speed === s.speed
                        ? 'bg-rf-cyan text-charcoal-950 border-rf-cyan-light shadow-sm'
                        : 'bg-charcoal-850 border-charcoal-750 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Epsilon Exploration Slider */}
            <div className="pt-2">
              <div className="flex items-center justify-between text-[11px] font-mono mb-2">
                <span className="text-slate-400">Exploration Rate (ε):</span>
                <span className="font-bold text-rf-amber-light">{(config.epsilon * 100).toFixed(0)}% exploration ({((1 - config.epsilon) * 100).toFixed(0)}% exploitation)</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.60"
                step="0.05"
                value={config.epsilon}
                onChange={(e) => onUpdateEpsilon(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-charcoal-950 rounded-lg appearance-none cursor-pointer accent-rf-amber"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
                <span>0.05 (High Exploit)</span>
                <span>0.20 (Standard ε)</span>
                <span>0.40</span>
                <span>0.60 (High Explore)</span>
              </div>
            </div>
          </div>

          {/* Live Status Summary Row */}
          <div className="grid grid-cols-4 gap-3">
            <div className="p-3 rounded-lg bg-charcoal-900 border border-charcoal-750">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Clock Time Slot</div>
              <div className="text-lg font-mono font-bold text-slate-100 mt-1">#{currentScan.time_slot}</div>
            </div>
            <div className="p-3 rounded-lg bg-charcoal-900 border border-charcoal-750">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Tuned Band</div>
              <div className="text-lg font-mono font-bold text-rf-green-light mt-1">BAND {currentScan.frequency_band}</div>
            </div>
            <div className="p-3 rounded-lg bg-charcoal-900 border border-charcoal-750">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Detection Power</div>
              <div className="text-lg font-mono font-bold text-slate-200 mt-1">{currentScan.signal_power.toFixed(2)} dB</div>
            </div>
            <div className="p-3 rounded-lg bg-charcoal-900 border border-charcoal-750">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Last Reward</div>
              <div className={`text-lg font-mono font-bold mt-1 ${currentScan.reward > 0 ? 'text-rf-green-light' : 'text-rf-red-light'}`}>
                {currentScan.reward > 0 ? `+${currentScan.reward.toFixed(1)}` : currentScan.reward.toFixed(1)}
              </div>
            </div>
          </div>
        </div>

        {/* Right 1 Col: Live Event Log Stream */}
        <div className="lg:col-span-1">
          <EventLog events={eventLogs} />
        </div>
      </div>
    </div>
  );
};
