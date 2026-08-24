import React from 'react';
import { 
  Radio, 
  Activity, 
  Cpu, 
  BarChart3, 
  PlaySquare, 
  ShieldCheck, 
  Sparkles
} from 'lucide-react';


export type TabId = 'command-center' | 'spectrum-intel' | 'ml-scheduler' | 'performance' | 'simulation';

interface SidebarProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  isRunning: boolean;
  demoMode: boolean;
  onToggleDemoMode: (enabled: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  isRunning,
  demoMode,
  onToggleDemoMode,
}) => {
  const navItems = [
    { id: 'command-center' as TabId, label: 'Command Center', icon: Radio, badge: 'PRIMARY' },
    { id: 'spectrum-intel' as TabId, label: 'Spectrum Intelligence', icon: Activity },
    { id: 'ml-scheduler' as TabId, label: 'ML Scheduler', icon: Cpu },
    { id: 'performance' as TabId, label: 'Performance', icon: BarChart3 },
    { id: 'simulation' as TabId, label: 'Simulation', icon: PlaySquare },
  ];

  return (
    <aside className="w-64 bg-charcoal-900 border-r border-charcoal-750/80 flex flex-col h-screen select-none sticky top-0 z-30">
      {/* Brand Header */}
      <div className="p-5 border-b border-charcoal-750/80 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-rf-green-dark via-charcoal-800 to-charcoal-900 border border-rf-green-border flex items-center justify-center shadow-glow-green">
            <span className="font-display font-black text-rf-green tracking-wider text-base">SS</span>
          </div>
          <div>
            <div className="font-display font-bold text-slate-100 tracking-wider text-sm flex items-center gap-1.5">
              SMART SCAN
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rf-green-bg text-rf-green-light border border-rf-green-border">EW</span>
            </div>
            <div className="text-[11px] text-slate-400 font-mono tracking-tight">Spectrum Intelligence</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[10px] font-mono font-semibold text-slate-500 uppercase tracking-wider">
          Operations
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                isActive
                  ? 'bg-charcoal-800 text-rf-green border border-rf-green-border shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-charcoal-850 border border-transparent'
              }`}
            >
              <div className="flex items-center space-x-3">
                <Icon className={`w-4 h-4 ${isActive ? 'text-rf-green' : 'text-slate-500'}`} />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span className={`text-[9px] font-mono px-1.5 py-0.2 rounded ${
                  isActive ? 'bg-rf-green/20 text-rf-green' : 'bg-charcoal-750 text-slate-400'
                }`}>
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}

        {/* Demo Mode Switcher Card */}
        <div className="pt-4 px-1">
          <div className="p-3 rounded-lg bg-charcoal-850/80 border border-charcoal-750">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-rf-amber" />
                <span className="text-[11px] font-mono font-semibold text-slate-300">Hackathon Demo</span>
              </div>
              <button
                onClick={() => onToggleDemoMode(!demoMode)}
                className={`relative inline-flex h-4 w-8 items-center rounded-full transition-colors ${
                  demoMode ? 'bg-rf-amber' : 'bg-charcoal-700'
                }`}
              >
                <span
                  className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                    demoMode ? 'translate-x-4' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>
            <p className="text-[10px] text-slate-400 leading-tight">
              {demoMode 
                ? 'Deterministic 10-step story showing discovery, hit & adaptive learning.' 
                : 'Free-running adaptive RF simulator.'}
            </p>
          </div>
        </div>
      </nav>

      {/* Footer System Status */}
      <div className="p-4 border-t border-charcoal-750/80 bg-charcoal-950/60">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">System Status</span>
          <span className="flex items-center gap-1.5 text-[10px] font-mono text-rf-green-light">
            <span className={`w-1.5 h-1.5 rounded-full ${isRunning ? 'bg-rf-green tactical-dot' : 'bg-rf-green'}`}></span>
            ONLINE
          </span>
        </div>
        <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
          <div className="flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-rf-cyan" />
            <span>Simulated RF Env</span>
          </div>
          <span className="text-[10px] text-slate-500">v2.4</span>
        </div>
      </div>
    </aside>
  );
};
