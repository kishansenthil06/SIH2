import { useState, useRef, useEffect } from 'react';
import { 
  Radio, 
  Activity, 
  Cpu, 
  BarChart3, 
  PlaySquare, 
  Sparkles, 
  Play, 
  Pause, 
  RotateCcw, 
  StepForward, 
  ChevronRight,
  LogOut,
  Home
} from 'lucide-react';

import type { UserProfile } from '../types/auth';

export type TabId = 'command-center' | 'spectrum-intel' | 'ml-scheduler' | 'performance' | 'simulation';

interface FloatingStaircaseNavProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  timeSlot: number;
  isRunning: boolean;
  demoMode: boolean;
  onToggleDemoMode: (enabled: boolean) => void;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onStep: () => void;
  currentUser?: UserProfile | null;
  onLogout?: () => void;
  onGoToLanding?: () => void;
}

export const FloatingStaircaseNav = ({
  activeTab,
  onTabChange,
  timeSlot,
  isRunning,
  demoMode,
  onToggleDemoMode,
  onStart,
  onPause,
  onReset,
  onStep,
  currentUser,
  onLogout,
  onGoToLanding,
}: FloatingStaircaseNavProps) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const hideTimerRef = useRef<number | null>(null);

  const steps = [
    {
      id: 'command-center' as TabId,
      stepNumber: '01',
      label: 'Command Center',
      subtext: '20-Band Spectrum, Live Dwells & Decisions',
      icon: Radio,
      indentClass: 'ml-0',
      animClass: 'staircase-step-1',
      badge: 'PRIMARY',
    },
    {
      id: 'spectrum-intel' as TabId,
      stepNumber: '02',
      label: 'Spectrum Intelligence',
      subtext: 'Distribution, 5000-Slot Scrubber & Receptivity',
      icon: Activity,
      indentClass: 'ml-6 sm:ml-8',
      animClass: 'staircase-step-2',
    },
    {
      id: 'ml-scheduler' as TabId,
      stepNumber: '03',
      label: 'ML Scheduler',
      subtext: 'Exploration vs Exploitation Policy & Rewards',
      icon: Cpu,
      indentClass: 'ml-12 sm:ml-16',
      animClass: 'staircase-step-3',
    },
    {
      id: 'performance' as TabId,
      stepNumber: '04',
      label: 'Performance',
      subtext: 'Traditional vs Adaptive Scheduler Benchmark',
      icon: BarChart3,
      indentClass: 'ml-18 sm:ml-24',
      animClass: 'staircase-step-4',
    },
    {
      id: 'simulation' as TabId,
      stepNumber: '05',
      label: 'Simulation',
      subtext: 'Speed Multipliers, Epsilon Tuning & Event Stream',
      icon: PlaySquare,
      indentClass: 'ml-24 sm:ml-32',
      animClass: 'staircase-step-5',
    },
  ];

  const handleMouseEnter = () => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    hideTimerRef.current = window.setTimeout(() => {
      setIsHovered(false);
    }, 280);
  };

  useEffect(() => {
    return () => {
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
    };
  }, []);

  return (
    <>
      {/* 1. FLOATING SMART SCAN LOGO (ALONE ON TOP-LEFT) WITH STAIRCASE HOVER */}
      <div
        className="fixed top-5 left-6 z-50 select-none"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {/* The Floating Logo Capsule */}
        <div
          onClick={() => onTabChange('command-center')}
          className="flex items-center gap-3 px-3.5 py-2 rounded-2xl bg-charcoal-900/90 backdrop-blur-xl border border-charcoal-700/80 shadow-2xl hover:border-rf-green-border transition-all duration-300 cursor-pointer group hover:shadow-glow-green"
        >
          {/* Logo Crest */}
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-rf-green-dark via-charcoal-800 to-charcoal-950 border border-rf-green-border flex items-center justify-center shadow-glow-green group-hover:scale-105 transition-transform">
            <span className="font-display font-black text-rf-green text-sm tracking-wider">SS</span>
          </div>

          {/* Logo Typography */}
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-display font-black text-sm text-slate-100 tracking-wider group-hover:text-rf-green-light transition-colors">
                SMART SCAN
              </span>
              <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-rf-green-bg text-rf-green border border-rf-green-border">
                EW
              </span>
            </div>
            <div className="text-[10px] text-slate-400 font-mono tracking-tight flex items-center gap-1.5">
              <span>Command Hub</span>
              <span className="w-1.5 h-1.5 rounded-full bg-rf-green tactical-dot"></span>
            </div>
          </div>
        </div>

        {/* 2. STAIRCASE STEPPING CASCADE MENU */}
        {isHovered && (
          <div className="pt-3 flex flex-col space-y-2 relative pointer-events-auto">
            {steps.map((step) => {
              const Icon = step.icon;
              const isActive = activeTab === step.id;

              return (
                <button
                  key={step.id}
                  onClick={() => {
                    onTabChange(step.id);
                    setIsHovered(false);
                  }}
                  className={`flex items-center gap-3.5 p-2.5 rounded-xl border backdrop-blur-2xl shadow-2xl transition-all duration-200 text-left w-80 group/step ${
                    step.indentClass
                  } ${step.animClass} ${
                    isActive
                      ? 'bg-charcoal-850/95 border-rf-green-border text-rf-green-light shadow-glow-green ring-1 ring-rf-green/40'
                      : 'bg-charcoal-900/90 hover:bg-charcoal-850/95 border-charcoal-750/90 text-slate-300 hover:border-slate-500'
                  }`}
                >
                  {/* Staircase Index Badge */}
                  <div className={`px-2 py-1 rounded-md font-mono text-[11px] font-bold shrink-0 ${
                    isActive ? 'bg-rf-green text-charcoal-950 shadow-sm' : 'bg-charcoal-800 text-slate-400 group-hover/step:text-slate-200'
                  }`}>
                    {step.stepNumber}
                  </div>

                  {/* Icon */}
                  <div className={`p-1.5 rounded-lg shrink-0 ${
                    isActive ? 'bg-rf-green-bg text-rf-green' : 'bg-charcoal-800 text-slate-400'
                  }`}>
                    <Icon className="w-4 h-4" />
                  </div>

                  {/* Text Details */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className={`font-mono font-bold text-xs truncate ${isActive ? 'text-rf-green-light' : 'text-slate-100'}`}>
                        {step.label}
                      </span>
                      {step.badge && (
                        <span className="text-[8px] font-mono px-1 rounded bg-rf-green/20 text-rf-green shrink-0">
                          {step.badge}
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-400 font-sans truncate mt-0.5 leading-tight">
                      {step.subtext}
                    </p>
                  </div>

                  <ChevronRight className={`w-3.5 h-3.5 shrink-0 opacity-0 group-hover/step:opacity-100 transition-opacity ${
                    isActive ? 'text-rf-green opacity-100' : 'text-slate-400'
                  }`} />
                </button>
              );
            })}

            {/* Quick Landing Page link at bottom of staircase */}
            {onGoToLanding && (
              <button
                onClick={() => {
                  onGoToLanding();
                  setIsHovered(false);
                }}
                className="ml-28 sm:ml-36 flex items-center gap-2 px-3 py-2 rounded-xl bg-charcoal-900/90 hover:bg-charcoal-800 border border-charcoal-750 text-slate-400 hover:text-slate-200 text-xs font-mono transition-all w-48"
              >
                <Home className="w-3.5 h-3.5 text-rf-cyan" />
                <span>Overview Landing</span>
              </button>
            )}
          </div>
        )}
      </div>

      {/* 3. COMPACT FLOATING CONTROLS PILL (TOP-RIGHT) */}
      <div className="fixed top-5 right-6 z-50 flex items-center space-x-3 select-none">
        {/* User Profile Capsule with Quick Menu */}
        {currentUser && (
          <div className="relative">
            <button
              onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
              className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-charcoal-900/90 backdrop-blur-xl border border-charcoal-700/80 shadow-2xl hover:border-rf-green-border transition-all text-xs"
            >
              <div className="w-5 h-5 rounded-md bg-rf-green-bg border border-rf-green text-rf-green font-mono font-bold text-[10px] flex items-center justify-center">
                {currentUser.avatarInitials}
              </div>
              <div className="text-left hidden sm:block">
                <div className="text-[11px] font-mono font-bold text-slate-200 leading-tight truncate max-w-[120px]">
                  {currentUser.name}
                </div>
                <div className="text-[9px] font-mono text-rf-cyan leading-none">
                  {currentUser.callsign}
                </div>
              </div>
            </button>

            {/* User Dropdown */}
            {isUserMenuOpen && (
              <div className="absolute right-0 top-full mt-2 w-64 rounded-2xl bg-charcoal-900/98 backdrop-blur-2xl border border-charcoal-700 shadow-2xl p-3 space-y-2 z-50 text-xs font-mono animate-in fade-in duration-150">
                <div className="pb-2 border-b border-charcoal-800">
                  <div className="font-bold text-slate-100">{currentUser.name}</div>
                  <div className="text-[10px] text-slate-400">{currentUser.role}</div>
                  <div className="text-[10px] text-rf-green-light mt-0.5">{currentUser.clearance}</div>
                </div>

                <div className="space-y-1">
                  {onGoToLanding && (
                    <button
                      onClick={() => {
                        setIsUserMenuOpen(false);
                        onGoToLanding();
                      }}
                      className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-charcoal-800 text-slate-300 text-left transition-colors"
                    >
                      <Home className="w-4 h-4 text-rf-cyan" />
                      <span>System Landing Overview</span>
                    </button>
                  )}

                  {onLogout && (
                    <button
                      onClick={() => {
                        setIsUserMenuOpen(false);
                        onLogout();
                      }}
                      className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-charcoal-800 text-rf-red-light text-left transition-colors"
                    >
                      <LogOut className="w-4 h-4 text-rf-red" />
                      <span>Switch User / Log Out</span>
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Hackathon Demo Toggle */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-charcoal-900/90 backdrop-blur-xl border border-charcoal-700/80 shadow-2xl text-xs">
          <Sparkles className={`w-3.5 h-3.5 ${demoMode ? 'text-rf-amber' : 'text-slate-400'}`} />
          <span className="text-[11px] font-mono text-slate-300 font-semibold">Demo Mode</span>
          <button
            onClick={() => onToggleDemoMode(!demoMode)}
            className={`relative inline-flex h-4 w-7 items-center rounded-full transition-colors ${
              demoMode ? 'bg-rf-amber' : 'bg-charcoal-700'
            }`}
          >
            <span
              className={`inline-block h-2.5 w-2.5 transform rounded-full bg-white transition-transform ${
                demoMode ? 'translate-x-3.5' : 'translate-x-0.5'
              }`}
            />
          </button>
        </div>

        {/* Current Time Slot Badge */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-charcoal-900/90 backdrop-blur-xl border border-charcoal-700/80 shadow-2xl text-xs">
          <Radio className="w-3.5 h-3.5 text-rf-green" />
          <div className="text-left">
            <div className="text-[8px] font-mono text-slate-400 uppercase leading-none">Time Slot</div>
            <div className="text-[11px] font-mono font-bold text-rf-green-light leading-tight">#{timeSlot}</div>
          </div>
        </div>

        {/* Simulation Playback Buttons */}
        <div className="flex items-center space-x-1.5 p-1 rounded-xl bg-charcoal-900/90 backdrop-blur-xl border border-charcoal-700/80 shadow-2xl">
          {isRunning ? (
            <button
              onClick={onPause}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-rf-amber/20 hover:bg-rf-amber/30 text-rf-amber border border-rf-amber/40 text-xs font-mono font-semibold transition-all shadow-glow-amber"
              title="Pause Simulation"
            >
              <Pause className="w-3.5 h-3.5" />
              <span>PAUSE</span>
            </button>
          ) : (
            <button
              onClick={onStart}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-rf-green/20 hover:bg-rf-green/30 text-rf-green border border-rf-green/40 text-xs font-mono font-semibold transition-all shadow-glow-green"
              title="Start Simulation"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>START</span>
            </button>
          )}

          <button
            onClick={onStep}
            disabled={isRunning}
            className="p-1.5 rounded-lg hover:bg-charcoal-800 text-slate-300 disabled:opacity-40 text-xs transition-colors"
            title="Single Step Forward"
          >
            <StepForward className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={onReset}
            className="p-1.5 rounded-lg hover:bg-charcoal-800 text-slate-300 text-xs transition-colors"
            title="Reset Simulation"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </>
  );
};
