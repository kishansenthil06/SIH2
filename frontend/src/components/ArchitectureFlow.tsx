import React from 'react';
import { Radio, Eye, CheckCircle2, Award, Cpu, RefreshCw } from 'lucide-react';


interface ArchitectureFlowProps {
  isRunning: boolean;
  currentBand: number;
  detectorActive: boolean;
  result: 'HIT' | 'MISS';
  reward: number;
  strategy: string;
}

export const ArchitectureFlow: React.FC<ArchitectureFlowProps> = ({
  isRunning,
  currentBand,
  detectorActive,
  result,
  reward,
  strategy,
}) => {
  const steps = [
    {
      id: 'env',
      label: 'RF ENVIRONMENT',
      sub: '20 Bands Dataset',
      icon: Radio,
      active: true,
      color: 'text-rf-cyan',
      borderColor: 'border-rf-cyan/30',
      bgColor: 'bg-rf-cyan/10',
    },
    {
      id: 'receiver',
      label: 'RECEIVER',
      sub: `Band ${currentBand} Tuned`,
      icon: RefreshCw,
      active: true,
      color: 'text-slate-300',
      borderColor: 'border-charcoal-600',
      bgColor: 'bg-charcoal-800',
    },
    {
      id: 'detector',
      label: 'DETECTOR',
      sub: detectorActive ? 'Active Signal' : 'Inactive Noise',
      icon: Eye,
      active: detectorActive,
      color: detectorActive ? 'text-rf-green-light' : 'text-slate-400',
      borderColor: detectorActive ? 'border-rf-green-border' : 'border-charcoal-700',
      bgColor: detectorActive ? 'bg-rf-green-bg' : 'bg-charcoal-850',
    },
    {
      id: 'evaluator',
      label: 'EVALUATOR',
      sub: `Ground Truth: ${result}`,
      icon: CheckCircle2,
      active: result === 'HIT',
      color: result === 'HIT' ? 'text-rf-green-light' : 'text-rf-red-light',
      borderColor: result === 'HIT' ? 'border-rf-green-border' : 'border-rf-red-border',
      bgColor: result === 'HIT' ? 'bg-rf-green-bg' : 'bg-rf-red-bg',
    },
    {
      id: 'reward',
      label: 'REWARD',
      sub: `${reward > 0 ? '+' : ''}${reward.toFixed(1)} Joules/Hit`,
      icon: Award,
      active: reward > 0,
      color: reward > 0 ? 'text-rf-amber' : 'text-slate-400',
      borderColor: reward > 0 ? 'border-rf-amber-border' : 'border-charcoal-700',
      bgColor: reward > 0 ? 'bg-rf-amber-bg' : 'bg-charcoal-850',
    },
    {
      id: 'scheduler',
      label: 'ML SCHEDULER',
      sub: `${strategy} Mode`,
      icon: Cpu,
      active: true,
      color: 'text-rf-green',
      borderColor: 'border-rf-green-border',
      bgColor: 'bg-rf-green-bg',
    },
  ];

  return (
    <div className="p-3.5 rounded-xl bg-charcoal-900/90 border border-charcoal-750/80 mb-5 relative overflow-hidden">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono uppercase font-bold text-slate-400 tracking-wider">
            Closed-Loop Decision & Feedback Architecture
          </span>
          <span className="text-[10px] text-slate-500 font-mono">| Reinforcement & Feedback Pipeline</span>
        </div>
        <div className="text-[10px] font-mono text-rf-cyan flex items-center gap-1">
          <RefreshCw className={`w-3 h-3 ${isRunning ? 'animate-spin' : ''}`} />
          <span>CLOSED LOOP SCANNER</span>
        </div>
      </div>

      <div className="grid grid-cols-6 gap-2 items-center relative">
        {steps.map((step) => {
          const Icon = step.icon;
          return (
            <React.Fragment key={step.id}>
              <div
                className={`p-2 rounded-lg border ${step.borderColor} ${step.bgColor} transition-all duration-300 relative`}
              >
                {/* Flow glow pulse */}
                {isRunning && (
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent animate-flow-right pointer-events-none rounded-lg" />
                )}
                <div className="flex items-center gap-2">
                  <Icon className={`w-3.5 h-3.5 ${step.color} shrink-0`} />
                  <div className="truncate">
                    <div className="text-[9px] font-mono font-bold text-slate-200 tracking-tight leading-tight truncate">
                      {step.label}
                    </div>
                    <div className={`text-[10px] font-mono font-semibold truncate ${step.color}`}>
                      {step.sub}
                    </div>
                  </div>
                </div>
              </div>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
