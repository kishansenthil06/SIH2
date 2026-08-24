import React from 'react';
import type { CurrentScan as CurrentScanType } from '../types/rf';
import { Radio, Eye, CheckCircle2, Award, Signal } from 'lucide-react';


interface CurrentScanProps {
  scan: CurrentScanType;
}

export const CurrentScan: React.FC<CurrentScanProps> = ({ scan }) => {
  const isHit = scan.result === 'HIT';
  const isDetectorActive = scan.detector_state === 'ACTIVE';

  // Normalize signal power for signal strength meter (from -5 dB to +15 dB)
  const signalPercent = Math.max(0, Math.min(100, ((scan.signal_power + 5) / 20) * 100));

  return (
    <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80 flex flex-col justify-between h-full relative overflow-hidden">
      {/* Top Banner */}
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-charcoal-750/80 mb-4">
          <div className="flex items-center space-x-2.5">
            <div className="p-1.5 rounded-md bg-rf-cyan-bg border border-rf-cyan-border text-rf-cyan">
              <Signal className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                CURRENT SCAN
              </h2>
              <p className="text-[11px] text-slate-400 font-sans">Receiver Dwell Telemetry</p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-charcoal-850 border border-charcoal-700">
            <span className="text-[10px] font-mono text-slate-400">SLOT</span>
            <span className="text-xs font-mono font-bold text-rf-green-light">#{scan.time_slot}</span>
          </div>
        </div>

        {/* Selected Frequency Banner */}
        <div className="p-3.5 rounded-lg bg-charcoal-850 border border-charcoal-700 mb-4 flex items-center justify-between">
          <div>
            <div className="text-[10px] font-mono uppercase text-slate-400 font-semibold">Tuned Center Frequency</div>
            <div className="text-xl font-mono font-black text-slate-100 flex items-baseline gap-2">
              BAND {scan.frequency_band}
              <span className="text-xs font-normal text-slate-400">
                ({(2.0 + (scan.frequency_band - 1) * 0.8).toFixed(1)} – {(2.8 + (scan.frequency_band - 1) * 0.8).toFixed(1)} GHz)
              </span>
            </div>
          </div>

          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-charcoal-800 to-charcoal-900 border border-charcoal-700 flex items-center justify-center text-rf-cyan">
            <Radio className="w-5 h-5" />
          </div>
        </div>

        {/* Observation Measurements Grid */}
        <div className="grid grid-cols-3 gap-2.5 mb-4">
          <div className="p-2.5 rounded-lg bg-charcoal-850/90 border border-charcoal-750">
            <div className="text-[9px] font-mono text-slate-400 uppercase">Signal Power</div>
            <div className={`text-base font-mono font-bold mt-0.5 ${scan.signal_power > 5.0 ? 'text-rf-green-light' : 'text-slate-300'}`}>
              {scan.signal_power.toFixed(2)} <span className="text-[10px] text-slate-400 font-normal">dB</span>
            </div>
          </div>

          <div className="p-2.5 rounded-lg bg-charcoal-850/90 border border-charcoal-750">
            <div className="text-[9px] font-mono text-slate-400 uppercase">Pulse Width</div>
            <div className="text-base font-mono font-bold text-slate-200 mt-0.5">
              {scan.pulse_width > 0 ? scan.pulse_width.toFixed(2) : '0.00'} <span className="text-[10px] text-slate-400 font-normal">µs</span>
            </div>
          </div>

          <div className="p-2.5 rounded-lg bg-charcoal-850/90 border border-charcoal-750">
            <div className="text-[9px] font-mono text-slate-400 uppercase">Angle of Arrival</div>
            <div className="text-base font-mono font-bold text-slate-200 mt-0.5">
              {scan.angle_of_arrival !== null ? `${scan.angle_of_arrival}°` : 'N/A'}
            </div>
          </div>
        </div>

        {/* Signal Strength Visual Meter */}
        <div className="mb-4 p-3 rounded-lg bg-charcoal-850/70 border border-charcoal-750">
          <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-1.5">
            <span>RF SIGNAL LEVEL</span>
            <span className="font-bold text-slate-200">{signalPercent.toFixed(0)}%</span>
          </div>
          <div className="w-full h-2 bg-charcoal-950 rounded-full overflow-hidden p-0.5 border border-charcoal-700">
            <div
              style={{ width: `${signalPercent}%` }}
              className={`h-full rounded-full transition-all duration-300 ${
                isHit
                  ? 'bg-gradient-to-r from-rf-green via-rf-green-light to-rf-cyan shadow-glow-green'
                  : 'bg-slate-500'
              }`}
            />
          </div>
        </div>
      </div>

      {/* Detector & Evaluator Verdict Row */}
      <div className="pt-3 border-t border-charcoal-750/80 space-y-2">
        <div className="grid grid-cols-2 gap-2">
          {/* Detector Verdict */}
          <div className={`p-2.5 rounded-lg border flex items-center justify-between ${
            isDetectorActive
              ? 'bg-rf-green-bg border-rf-green-border text-rf-green-light'
              : 'bg-charcoal-850 border-charcoal-750 text-slate-400'
          }`}>
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4" />
              <div>
                <div className="text-[9px] font-mono uppercase text-slate-400">Detector</div>
                <div className="text-xs font-mono font-bold">{scan.detector_state}</div>
              </div>
            </div>
          </div>

          {/* Evaluator Verdict */}
          <div className={`p-2.5 rounded-lg border flex items-center justify-between ${
            isHit
              ? 'bg-rf-green-bg border-rf-green-border text-rf-green-light'
              : 'bg-rf-red-bg border-rf-red-border text-rf-red-light'
          }`}>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" />
              <div>
                <div className="text-[9px] font-mono uppercase text-slate-400">Evaluator</div>
                <div className="text-xs font-mono font-bold">{scan.result}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Reward Result Card */}
        <div className="p-2.5 rounded-lg bg-charcoal-850 border border-charcoal-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-rf-amber" />
            <span className="text-[11px] font-mono text-slate-300">Observation Reward</span>
          </div>
          <span className={`font-mono font-bold text-sm ${scan.reward > 0 ? 'text-rf-green-light' : 'text-rf-red-light'}`}>
            {scan.reward > 0 ? `+${scan.reward.toFixed(1)}` : scan.reward.toFixed(1)} J
          </span>
        </div>
      </div>
    </div>
  );
};
