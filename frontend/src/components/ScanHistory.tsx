import React, { useState } from 'react';
import type { ScanHistoryRecord } from '../types/rf';
import { History, ChevronLeft, ChevronRight } from 'lucide-react';


interface ScanHistoryProps {
  history: ScanHistoryRecord[];
}

export const ScanHistory: React.FC<ScanHistoryProps> = ({ history }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 8;

  const totalPages = Math.ceil(history.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentItems = history.slice(startIndex, startIndex + itemsPerPage);

  return (
    <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-charcoal-750/80">
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 rounded-md bg-charcoal-800 border border-charcoal-700 text-slate-300">
            <History className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              SCAN HISTORY & TELEMETRY LOG
            </h2>
            <p className="text-[11px] text-slate-400 font-sans">Sequential Observation Audit Log</p>
          </div>
        </div>

        {/* Pagination Controls */}
        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="text-slate-400 text-[11px]">
            Page {currentPage} of {Math.max(1, totalPages)} ({history.length} scans)
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="p-1 rounded bg-charcoal-850 hover:bg-charcoal-800 disabled:opacity-30 text-slate-300 border border-charcoal-700 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage >= totalPages}
            className="p-1 rounded bg-charcoal-850 hover:bg-charcoal-800 disabled:opacity-30 text-slate-300 border border-charcoal-700 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead>
            <tr className="border-b border-charcoal-750/80 text-[10px] text-slate-400 uppercase tracking-wider bg-charcoal-850/50">
              <th className="py-2.5 px-3">Time Slot</th>
              <th className="py-2.5 px-3">Frequency Band</th>
              <th className="py-2.5 px-3">Signal Power</th>
              <th className="py-2.5 px-3">Pulse Width</th>
              <th className="py-2.5 px-3">Angle of Arrival</th>
              <th className="py-2.5 px-3">Detector</th>
              <th className="py-2.5 px-3">Evaluator Result</th>
              <th className="py-2.5 px-3 text-right">Reward</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-charcoal-800/60">
            {currentItems.map((rec) => {
              const isHit = rec.result === 'HIT';
              const isActive = rec.detector_state === 'ACTIVE';

              return (
                <tr key={rec.id} className="hover:bg-charcoal-850/60 transition-colors">
                  <td className="py-2.5 px-3 text-slate-300 font-bold">#{rec.time_slot}</td>
                  <td className="py-2.5 px-3 font-semibold text-slate-100">
                    <span className="px-2 py-0.5 rounded bg-charcoal-800 border border-charcoal-700">
                      BAND {rec.frequency_band}
                    </span>
                  </td>
                  <td className={`py-2.5 px-3 font-bold ${rec.signal_power > 5.0 ? 'text-rf-green-light' : 'text-slate-400'}`}>
                    {rec.signal_power.toFixed(2)} dB
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">
                    {rec.pulse_width > 0 ? `${rec.pulse_width.toFixed(2)} µs` : '0.00'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">
                    {rec.angle_of_arrival !== null ? `${rec.angle_of_arrival}°` : '—'}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold ${
                        isActive
                          ? 'bg-rf-green-bg text-rf-green-light border border-rf-green-border'
                          : 'bg-charcoal-800 text-slate-400'
                      }`}
                    >
                      {rec.detector_state}
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold ${
                        isHit
                          ? 'bg-rf-green-bg text-rf-green-light border border-rf-green-border shadow-glow-green'
                          : 'bg-rf-red-bg text-rf-red-light border border-rf-red-border'
                      }`}
                    >
                      {rec.result}
                    </span>
                  </td>
                  <td className={`py-2.5 px-3 text-right font-bold ${rec.reward > 0 ? 'text-rf-green-light' : 'text-rf-red-light'}`}>
                    {rec.reward > 0 ? `+${rec.reward.toFixed(1)}` : rec.reward.toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
