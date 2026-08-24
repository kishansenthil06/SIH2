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
  Database,
  ChevronDown,
  ArrowRight
} from 'lucide-react';

export type TabId = 'command-center' | 'spectrum-intel' | 'ml-scheduler' | 'performance' | 'simulation';

interface TopNavbarProps {
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
  /** Clicking the brand goes home. Optional so this component still works
   *  standalone; see FloatingStaircaseNav, which is the nav actually rendered. */
  onGoToLanding?: () => void;
}

export const TopNavbar = ({
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
  onGoToLanding,
}: TopNavbarProps) => {
  const [isLogoHovered, setIsLogoHovered] = useState(false);
  const dropdownTimeoutRef = useRef<number | null>(null);

  const navItems = [
    { 
      id: 'command-center' as TabId, 
      label: 'Command Center', 
      desc: '20-Band Live RF Spectrum, Dwell Telemetry & Decisions',
      icon: Radio, 
      stepClass: 'step-item-1',
      badge: 'PRIMARY' 
    },
    { 
      id: 'spectrum-intel' as TabId, 
      label: 'Spectrum Intelligence', 
      desc: 'Full-Band Distribution, Time Scrubber & Band Receptivity',
      icon: Activity, 
      stepClass: 'step-item-2' 
    },
    { 
      id: 'ml-scheduler' as TabId, 
      label: 'ML Scheduler', 
      desc: 'Exploration vs Exploitation Policy & Reward Learning',
      icon: Cpu, 
      stepClass: 'step-item-3' 
    },
    { 
      id: 'performance' as TabId, 
      label: 'Performance', 
      desc: 'Traditional Open-Loop vs Adaptive Scheduler Benchmark',
      icon: BarChart3, 
      stepClass: 'step-item-4' 
    },
    { 
      id: 'simulation' as TabId, 
      label: 'Simulation', 
      desc: 'Speed Multipliers, Epsilon Tuning & Real-Time Event Stream',
      icon: PlaySquare, 
      stepClass: 'step-item-5' 
    },
  ];

  const handleMouseEnter = () => {
    if (dropdownTimeoutRef.current) {
      clearTimeout(dropdownTimeoutRef.current);
      dropdownTimeoutRef.current = null;
    }
    setIsLogoHovered(true);
  };

  const handleMouseLeave = () => {
    dropdownTimeoutRef.current = window.setTimeout(() => {
      setIsLogoHovered(false);
    }, 250);
  };

  useEffect(() => {
    return () => {
      if (dropdownTimeoutRef.current) clearTimeout(dropdownTimeoutRef.current);
    };
  }, []);

  return (
    <header className="sticky top-0 z-40 bg-charcoal-900/95 backdrop-blur-md border-b border-charcoal-750/80 shadow-panel">
      {/* Upper Primary Navigation Bar */}
      <div className="px-6 py-3 flex items-center justify-between gap-4">
        {/* Left: SMART SCAN Logo with Stepwise Hover Dropdown */}
        <div 
          className="relative"
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        >
          <div 
            className="flex items-center space-x-3 cursor-pointer group py-1 pr-2 rounded-xl transition-all focus:outline-none focus:ring-2 focus:ring-rf-green/60"
            role="link"
            tabIndex={0}
            aria-label="Smart Scan - go to landing page"
            onClick={() => (onGoToLanding ? onGoToLanding() : onTabChange('command-center'))}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onGoToLanding ? onGoToLanding() : onTabChange('command-center');
              }
            }}
          >
            {/* Logo Icon Badge */}
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-rf-green-dark via-charcoal-800 to-charcoal-900 border border-rf-green-border flex items-center justify-center shadow-glow-green group-hover:scale-105 transition-transform">
              <span className="font-display font-black text-rf-green tracking-wider text-base">SS</span>
            </div>

            {/* Brand Titles */}
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-display font-bold text-slate-100 tracking-wider text-sm group-hover:text-rf-green-light transition-colors">
                  SMART SCAN
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-rf-green-bg text-rf-green border border-rf-green-border">
                  EW COMMAND
                </span>
                <ChevronDown className={`w-3.5 h-3.5 text-slate-400 group-hover:text-rf-green transition-transform duration-200 ${isLogoHovered ? 'rotate-180 text-rf-green' : ''}`} />
              </div>
              <div className="text-[10px] text-slate-400 font-mono tracking-tight flex items-center gap-1.5">
                <span>AI-Driven Spectrum Intelligence</span>
                <span className="w-1 h-1 rounded-full bg-rf-green tactical-dot"></span>
              </div>
            </div>
          </div>

          {/* STEPWISE CASCADE DROPDOWN MENU */}
          {isLogoHovered && (
            <div className="absolute left-0 top-full mt-2 w-96 rounded-2xl bg-charcoal-900/98 backdrop-blur-xl border border-charcoal-700 shadow-2xl p-3 space-y-1.5 z-50 animate-in fade-in duration-150">
              <div className="px-3 py-1.5 text-[10px] font-mono font-bold uppercase tracking-wider text-slate-400 border-b border-charcoal-750 flex items-center justify-between">
                <span>TACTICAL OPERATIONS MODULES</span>
                <span className="text-rf-green font-normal">Select View</span>
              </div>

              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      onTabChange(item.id);
                      setIsLogoHovered(false);
                    }}
                    className={`w-full flex items-start gap-3 p-2.5 rounded-xl text-left transition-all ${item.stepClass} ${
                      isActive
                        ? 'bg-charcoal-800/90 text-rf-green-light border border-rf-green-border shadow-glow-green'
                        : 'hover:bg-charcoal-800/60 text-slate-300 border border-transparent hover:border-charcoal-700'
                    }`}
                  >
                    <div className={`p-2 rounded-lg shrink-0 ${
                      isActive ? 'bg-rf-green-bg text-rf-green border border-rf-green-border' : 'bg-charcoal-800 text-slate-400'
                    }`}>
                      <Icon className="w-4 h-4" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <span className="font-mono font-bold text-xs text-slate-100 flex items-center gap-2">
                          {item.label}
                          {item.badge && (
                            <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-rf-green/20 text-rf-green">
                              {item.badge}
                            </span>
                          )}
                        </span>
                        <ArrowRight className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 text-rf-green transition-opacity" />
                      </div>
                      <p className="text-[10px] text-slate-400 font-sans mt-0.5 leading-snug truncate">
                        {item.desc}
                      </p>
                    </div>
                  </button>
                );
              })}

              <div className="pt-2 mt-1 border-t border-charcoal-750/80 px-2 flex items-center justify-between text-[10px] font-mono text-slate-400">
                <span>Mode: <strong>Simulation / Prototype</strong></span>
                <span className="text-rf-cyan">20 Frequency Channels</span>
              </div>
            </div>
          )}
        </div>

        {/* Center: Top Navigation Tabs */}
        <nav className="hidden lg:flex items-center space-x-1.5 bg-charcoal-850/80 p-1.5 rounded-xl border border-charcoal-750">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                  isActive
                    ? 'bg-charcoal-750 text-rf-green-light border border-rf-green-border shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-charcoal-800 border border-transparent'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-rf-green' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right: Telemetry Badges & Simulation Playback Controls */}
        <div className="flex items-center space-x-3">
          {/* Hackathon Demo Toggle */}
          <div className="flex items-center space-x-2 px-2.5 py-1.5 rounded-lg bg-charcoal-850 border border-charcoal-750 text-xs">
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

          {/* Time Slot Badge */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-charcoal-850 border border-charcoal-750 text-xs">
            <Radio className="w-3.5 h-3.5 text-rf-green" />
            <div className="text-left">
              <div className="text-[8px] font-mono text-slate-400 uppercase leading-none">Time Slot</div>
              <div className="text-[11px] font-mono font-bold text-rf-green-light leading-tight">#{timeSlot}</div>
            </div>
          </div>

          {/* Environment Badge */}
          <div className="hidden xl:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-charcoal-850 border border-charcoal-750 text-xs">
            <Database className="w-3.5 h-3.5 text-rf-cyan" />
            <div className="text-left">
              <div className="text-[8px] font-mono text-slate-400 uppercase leading-none">RF Source</div>
              <div className="text-[11px] font-mono font-semibold text-slate-200 leading-tight">100k Dataset</div>
            </div>
          </div>

          {/* Simulation Controls */}
          <div className="flex items-center space-x-1 bg-charcoal-850 p-1 rounded-lg border border-charcoal-750">
            {isRunning ? (
              <button
                onClick={onPause}
                className="flex items-center gap-1 px-2.5 py-1 rounded bg-rf-amber/20 hover:bg-rf-amber/30 text-rf-amber border border-rf-amber/40 text-xs font-mono font-semibold transition-all"
                title="Pause Simulation"
              >
                <Pause className="w-3.5 h-3.5" />
                <span>PAUSE</span>
              </button>
            ) : (
              <button
                onClick={onStart}
                className="flex items-center gap-1 px-2.5 py-1 rounded bg-rf-green/20 hover:bg-rf-green/30 text-rf-green border border-rf-green/40 text-xs font-mono font-semibold transition-all"
                title="Start Simulation"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>START</span>
              </button>
            )}

            <button
              onClick={onStep}
              disabled={isRunning}
              className="p-1.5 rounded hover:bg-charcoal-700 text-slate-300 disabled:opacity-40 text-xs transition-colors"
              title="Single Step"
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
      </div>
    </header>
  );
};
