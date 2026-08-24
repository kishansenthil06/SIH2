import React from 'react';
import type { LucideIcon } from 'lucide-react';


interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  variant?: 'green' | 'cyan' | 'amber' | 'neutral';
  trend?: string;
  isGlowing?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  variant = 'neutral',
  trend,
  isGlowing = false,
}) => {
  const variantStyles = {
    green: {
      border: 'border-rf-green-border',
      bg: 'bg-charcoal-850/90',
      iconBg: 'bg-rf-green-bg text-rf-green',
      valueText: 'text-rf-green-light',
      glow: 'shadow-glow-green',
    },
    cyan: {
      border: 'border-rf-cyan-border',
      bg: 'bg-charcoal-850/90',
      iconBg: 'bg-rf-cyan-bg text-rf-cyan',
      valueText: 'text-rf-cyan-light',
      glow: 'shadow-glow-cyan',
    },
    amber: {
      border: 'border-rf-amber-border',
      bg: 'bg-charcoal-850/90',
      iconBg: 'bg-rf-amber-bg text-rf-amber',
      valueText: 'text-rf-amber-light',
      glow: 'shadow-glow-amber',
    },
    neutral: {
      border: 'border-charcoal-750',
      bg: 'bg-charcoal-850/80',
      iconBg: 'bg-charcoal-750 text-slate-400',
      valueText: 'text-slate-100',
      glow: '',
    },
  }[variant];

  return (
    <div
      className={`p-4 rounded-xl border ${variantStyles.border} ${variantStyles.bg} ${
        isGlowing ? variantStyles.glow : ''
      } transition-all duration-200 relative overflow-hidden`}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-semibold mb-1">
            {title}
          </div>
          <div className={`text-2xl font-mono font-bold tracking-tight ${variantStyles.valueText}`}>
            {value}
          </div>
          {subtitle && (
            <div className="text-[11px] text-slate-400 font-sans mt-0.5 flex items-center gap-1.5">
              {subtitle}
            </div>
          )}
        </div>
        <div className={`p-2 rounded-lg ${variantStyles.iconBg}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
      {trend && (
        <div className="mt-2 pt-2 border-t border-charcoal-750/50 flex items-center justify-between text-[10px] font-mono text-slate-400">
          <span>{trend}</span>
        </div>
      )}
    </div>
  );
};
