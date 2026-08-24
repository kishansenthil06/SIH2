import React from 'react';
import type { BandState } from '../types/rf';
import { X, Radio, Activity, Target } from 'lucide-react';


interface BandDetailModalProps {
  band: BandState | null;
  onClose: () => void;
}

export const BandDetailModal: React.FC<BandDetailModalProps> = ({ band, onClose }) => {
  if (!band) return null;

  const fMin = (2.0 + (band.band - 1) * 0.8).toFixed(1);
  const fMax = (2.8 + (band.band - 1) * 0.8).toFixed(1);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-lg bg-charcoal-900 border border-charcoal-700 rounded-2xl shadow-2xl p-6 relative overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4 border-b border-charcoal-750">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-rf-green-bg border border-rf-green-border text-rf-green">
              <Radio className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-mono font-bold text-base text-slate-100 flex items-center gap-2">
                FREQUENCY BAND {band.band}
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                  band.priority === 'VERY_HIGH'
                    ? 'bg-rf-amber-bg text-rf-amber-light border border-rf-amber-border'
                    : 'bg-charcoal-800 text-slate-300'
                }`}>
                  {band.priority} PRIORITY
                </span>
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                Spectral Range: {fMin} GHz – {fMax} GHz (800 MHz Channel BW)
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-charcoal-800 hover:bg-charcoal-750 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body Stats Grid */}
        <div className="py-5 space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-charcoal-850 border border-charcoal-750">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Current Power</div>
              <div className={`text-lg font-mono font-bold mt-1 ${band.signal_power > 5 ? 'text-rf-green-light' : 'text-slate-200'}`}>
                {band.signal_power.toFixed(2)} dB
              </div>
            </div>

            <div className="p-3 rounded-lg bg-charcoal-850 border border-charcoal-750">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Pulse Width</div>
              <div className="text-lg font-mono font-bold text-slate-200 mt-1">
                {band.pulse_width > 0 ? `${band.pulse_width.toFixed(2)} µs` : '0.00 µs'}
              </div>
            </div>

            <div className="p-3 rounded-lg bg-charcoal-850 border border-charcoal-750">
              <div className="text-[10px] font-mono text-slate-400 uppercase">Angle of Arrival</div>
              <div className="text-lg font-mono font-bold text-slate-200 mt-1">
                {band.angle_of_arrival !== null ? `${band.angle_of_arrival}°` : 'N/A'}
              </div>
            </div>
          </div>

          {/* Historical Intelligence */}
          <div className="p-4 rounded-xl bg-charcoal-850/80 border border-charcoal-750 space-y-3">
            <div className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-rf-cyan" />
              <span>Historical Telemetry & Hit Statistics</span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-xs font-mono">
              <div className="p-2 rounded bg-charcoal-900 border border-charcoal-800">
                <span className="text-[10px] text-slate-400">Total Dwells:</span>
                <div className="font-bold text-slate-100 mt-0.5">{band.scans} scans</div>
              </div>
              <div className="p-2 rounded bg-charcoal-900 border border-charcoal-800">
                <span className="text-[10px] text-slate-400">Confirmed Hits:</span>
                <div className="font-bold text-rf-green-light mt-0.5">{band.hits} hits</div>
              </div>
              <div className="p-2 rounded bg-charcoal-900 border border-charcoal-800">
                <span className="text-[10px] text-slate-400">Hit Rate:</span>
                <div className="font-bold text-rf-cyan-light mt-0.5">{band.hit_rate}%</div>
              </div>
            </div>
          </div>

          {/* Tactical Recommendation */}
          <div className="p-3.5 rounded-lg bg-charcoal-850/60 border border-charcoal-750 text-xs">
            <div className="text-[10px] font-mono text-slate-400 uppercase font-semibold mb-1 flex items-center gap-1">
              <Target className="w-3 h-3 text-rf-amber" />
              <span>Tactical Assessment</span>
            </div>
            <p className="text-slate-300 font-sans text-xs leading-relaxed">
              {band.hit_rate > 50
                ? `High-yield operational channel with ${band.hits} verified emitter burst detections. Recommended for high-priority interleaved dwell.`
                : band.hit_rate > 20
                ? `Moderate activity channel. Periodic exploration scans recommended to track frequency agility or hopping emitter dynamics.`
                : `Low background channel. Serves primarily as baseline noise floor monitoring with standard exploration periodicity.`}
            </p>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="pt-3 border-t border-charcoal-750 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-charcoal-800 hover:bg-charcoal-700 text-slate-200 text-xs font-mono font-semibold transition-colors"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
};
