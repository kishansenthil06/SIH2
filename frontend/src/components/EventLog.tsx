import React from 'react';
import type { EventLogItem } from '../types/simulation';
import { Terminal } from 'lucide-react';


interface EventLogProps {
  events: EventLogItem[];
}

export const EventLog: React.FC<EventLogProps> = ({ events }) => {
  return (
    <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 pb-3 border-b border-charcoal-750/80">
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 rounded-md bg-charcoal-800 border border-charcoal-700 text-slate-300">
            <Terminal className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              REAL-TIME EVENT STREAM
            </h3>
            <p className="text-[11px] text-slate-400 font-sans">Live Pipeline Telemetry & Decisions</p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] font-mono text-rf-green-light">
          <span className="w-1.5 h-1.5 rounded-full bg-rf-green tactical-dot"></span>
          <span>STREAM ACTIVE</span>
        </div>
      </div>

      {/* Terminal Output */}
      <div className="flex-1 bg-charcoal-950/90 rounded-lg p-3 border border-charcoal-800 font-mono text-xs overflow-y-auto space-y-2 max-h-96">
        {events.length === 0 ? (
          <div className="text-slate-500 text-center py-8 text-[11px]">
            Waiting for simulation events... Click START to begin dwell stream.
          </div>
        ) : (
          events.map((evt) => {
            const levelColor = {
              info: 'text-slate-400',
              success: 'text-rf-green-light font-semibold',
              warning: 'text-rf-amber font-semibold',
              alert: 'text-rf-cyan-light font-bold',
            }[evt.level];

            const typeBadge = {
              decision: 'bg-rf-green-bg text-rf-green border-rf-green-border',
              scan: 'bg-charcoal-800 text-slate-300 border-charcoal-700',
              detector: 'bg-rf-cyan-bg text-rf-cyan border-rf-cyan-border',
              evaluator: 'bg-rf-amber-bg text-rf-amber border-rf-amber-border',
              learning: 'bg-charcoal-750 text-slate-200 border-charcoal-600',
              system: 'bg-charcoal-800 text-slate-400 border-charcoal-700',
            }[evt.type];

            return (
              <div key={evt.id} className="flex items-start space-x-2 text-[11px] leading-relaxed">
                <span className="text-slate-600 shrink-0">[{evt.timestamp}]</span>
                <span className={`px-1.5 py-0.2 text-[9px] uppercase rounded border shrink-0 ${typeBadge}`}>
                  {evt.type}
                </span>
                <span className={`${levelColor} break-words`}>{evt.message}</span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
