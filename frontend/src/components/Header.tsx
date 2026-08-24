import React from 'react';
import { Play, Pause, RotateCcw, StepForward, Radio, Database, Sparkles } from 'lucide-react';

interface HeaderProps {
  timeSlot: number;
  isRunning: boolean;
  demoMode: boolean;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onStep: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  timeSlot,
  isRunning,
  demoMode,
  onStart,
  onPause,
  onReset,
  onStep,
}) => {
  return (
    <header className="h-16 bg-charcoal-900/90 backdrop-blur border-b border-charcoal-750/80 px-6 flex items-center justify-between sticky top-0 z-20">
      {/* Title & Subtitle */}
      <div className="flex items-center space-x-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="font-display font-bold text-base text-slate-100 tracking-wide uppercase">
              SMART SCAN COMMAND CENTER
            </h1>
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono bg-rf-green-bg text-rf-green border border-rf-green-border">
              <span className={`w-1.5 h-1.5 rounded-full bg-rf-green ${isRunning ? 'tactical-dot' : ''}`}></span>
              RF SIMULATION ONLINE
            </span>
            {demoMode && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono bg-rf-amber-bg text-rf-amber border border-rf-amber-border animate-pulse">
                <Sparkles className="w-3 h-3" />
                DEMO MODE ACTIVE
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-400 font-sans tracking-tight">
            AI-Driven Electronic Warfare Spectrum Intelligence & Adaptive Scan Strategy
          </p>
        </div>
      </div>

      {/* Center/Right Status Badges & Controls */}
      <div className="flex items-center space-x-4">
        {/* Environment Badge */}
        <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-charcoal-850 border border-charcoal-750 text-xs">
          <Database className="w-3.5 h-3.5 text-rf-cyan" />
          <div className="text-left">
            <div className="text-[9px] font-mono text-slate-400 uppercase leading-none">Environment</div>
            <div className="text-[11px] font-mono font-semibold text-slate-200 leading-tight">Temporary RF Dataset</div>
          </div>
        </div>

        {/* Current Time Slot Badge */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-charcoal-850 border border-charcoal-750 text-xs">
          <Radio className="w-3.5 h-3.5 text-rf-green" />
          <div className="text-left">
            <div className="text-[9px] font-mono text-slate-400 uppercase leading-none">Current Time Slot</div>
            <div className="text-[12px] font-mono font-bold text-rf-green-light leading-tight">#{timeSlot}</div>
          </div>
        </div>

        {/* Simulation Control Buttons */}
        <div className="flex items-center space-x-1.5 bg-charcoal-850 p-1 rounded-lg border border-charcoal-750">
          {isRunning ? (
            <button
              onClick={onPause}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded bg-rf-amber/20 hover:bg-rf-amber/30 text-rf-amber border border-rf-amber/40 text-xs font-mono font-semibold transition-all"
              title="Pause Simulation"
            >
              <Pause className="w-3.5 h-3.5" />
              <span>PAUSE</span>
            </button>
          ) : (
            <button
              onClick={onStart}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded bg-rf-green/20 hover:bg-rf-green/30 text-rf-green border border-rf-green/40 text-xs font-mono font-semibold transition-all"
              title="Start Simulation"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>START</span>
            </button>
          )}

          <button
            onClick={onStep}
            disabled={isRunning}
            className="p-1.5 rounded hover:bg-charcoal-700 text-slate-300 disabled:opacity-40 disabled:hover:bg-transparent text-xs transition-colors"
            title="Single Step Forward"
          >
            <StepForward className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={onReset}
            className="p-1.5 rounded hover:bg-charcoal-700 text-slate-300 text-xs transition-colors"
            title="Reset Simulation"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </header>
  );
};
