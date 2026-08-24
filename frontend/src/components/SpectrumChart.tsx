import React, { useState } from 'react';
import type { BandState } from '../types/rf';
import { Radio, ArrowUpRight } from 'lucide-react';


interface SpectrumChartProps {
  bands: BandState[];
  selectedBand: number;
  onSelectBand: (band: BandState) => void;
}

export const SpectrumChart: React.FC<SpectrumChartProps> = ({
  bands,
  selectedBand,
  onSelectBand,
}) => {
  const [hoveredBand, setHoveredBand] = useState<BandState | null>(null);

  // Power scale: from -5 dB to +15 dB
  const minDb = -5;
  const maxDb = 15;

  return (
    <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80 relative">
      {/* Header with Spectrum Legend */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-1.5 rounded-md bg-rf-green-bg border border-rf-green-border text-rf-green">
            <Radio className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              REAL-TIME RF SPECTRUM
            </h2>
            <p className="text-[11px] text-slate-400 font-sans">
              20-Band Frequency Grid (2.0 GHz – 18.0 GHz ESM Surveillance Band)
            </p>
          </div>
        </div>

        {/* Tactical Legend */}
        <div className="flex items-center space-x-4 text-[10px] font-mono text-slate-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-rf-green shadow-glow-green border border-rf-green-light"></span>
            <span>Active / Hit</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-charcoal-700 border border-rf-green"></span>
            <span>Current Scan</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-charcoal-750 border border-rf-cyan"></span>
            <span>Recently Scanned</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-charcoal-700"></span>
            <span>Inactive Noise</span>
          </div>
        </div>
      </div>

      {/* Spectrum Display Area */}
      <div className="relative h-64 bg-charcoal-950/80 rounded-lg p-4 border border-charcoal-800 flex flex-col justify-between overflow-hidden">
        {/* Background Grid Lines (dB Power Level markers) */}
        <div className="absolute inset-x-4 inset-y-4 flex flex-col justify-between pointer-events-none opacity-20">
          {[15, 10, 5, 0, -5].map((db) => (
            <div key={db} className="w-full border-b border-dashed border-slate-500 relative">
              <span className="absolute -left-1 -top-2.5 text-[9px] font-mono text-slate-400">
                {db > 0 ? `+${db}` : db} dB
              </span>
            </div>
          ))}
        </div>

        {/* 5.0 dB Detection Threshold Line */}
        <div 
          className="absolute inset-x-4 border-b-2 border-rf-amber/50 border-dotted pointer-events-none z-10"
          style={{ bottom: `${((5 - minDb) / (maxDb - minDb)) * 100}%` }}
        >
          <span className="absolute right-2 -top-3 text-[9px] font-mono text-rf-amber font-semibold bg-charcoal-950/90 px-1 rounded">
            Detection Threshold: 5.0 dB
          </span>
        </div>

        {/* 20 Frequency Bars */}
        <div className="relative z-10 h-full flex items-end justify-between gap-1.5 pt-4 pb-1">
          {bands.map((b) => {
            const isSelected = b.band === selectedBand;
            const isDetectorActive = b.detector_state === 'ACTIVE' || b.signal_power > 5.0;
            const isRecent = b.is_recently_scanned;
            
            // Normalized height (clamp between 6% and 98%)
            const normHeight = Math.max(6, Math.min(98, ((b.signal_power - minDb) / (maxDb - minDb)) * 100));

            return (
              <div
                key={b.band}
                onMouseEnter={() => setHoveredBand(b)}
                onMouseLeave={() => setHoveredBand(null)}
                onClick={() => onSelectBand(b)}
                className="flex-1 h-full flex flex-col justify-end items-center group cursor-pointer relative"
              >
                {/* Visual Top Marker on Active/Selected */}
                {isSelected && (
                  <div className="w-2 h-2 mb-1 rounded-full bg-rf-green tactical-dot shadow-glow-green" />
                )}

                {/* The Bar */}
                <div
                  style={{ height: `${normHeight}%` }}
                  className={`w-full rounded-t transition-all duration-200 relative ${
                    isDetectorActive
                      ? 'bg-gradient-to-t from-rf-green-dark via-rf-green to-rf-green-light shadow-glow-green'
                      : isRecent
                      ? 'bg-gradient-to-t from-charcoal-800 to-rf-cyan/60 border-t border-rf-cyan'
                      : 'bg-gradient-to-t from-charcoal-850 to-charcoal-700 hover:to-slate-500'
                  } ${
                    isSelected
                      ? 'ring-2 ring-rf-green ring-offset-1 ring-offset-charcoal-950'
                      : ''
                  }`}
                >
                  {/* ML Priority Tag Indicator */}
                  {b.priority === 'VERY_HIGH' && (
                    <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-rf-amber animate-ping" />
                  )}
                </div>

                {/* Band Label */}
                <div className={`text-[10px] font-mono mt-1.5 font-semibold transition-colors ${
                  isSelected ? 'text-rf-green-light font-bold' : 'text-slate-400 group-hover:text-slate-200'
                }`}>
                  F{b.band}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Floating Detailed Hover Card */}
      {hoveredBand && (
        <div className="mt-3 p-3 rounded-lg bg-charcoal-850 border border-charcoal-700 shadow-xl flex items-center justify-between text-xs animate-in fade-in duration-150">
          <div className="flex items-center space-x-4">
            <div className="flex items-center gap-2">
              <span className="font-mono font-bold text-slate-100 text-sm">BAND {hoveredBand.band}</span>
              <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                hoveredBand.detector_state === 'ACTIVE'
                  ? 'bg-rf-green-bg text-rf-green-light border border-rf-green-border'
                  : 'bg-charcoal-750 text-slate-400'
              }`}>
                {hoveredBand.detector_state}
              </span>
            </div>
            <div className="text-slate-300 font-mono">
              Power: <span className="font-bold text-slate-100">{hoveredBand.signal_power.toFixed(2)} dB</span>
            </div>
            <div className="text-slate-300 font-mono">
              Pulse Width: <span className="font-bold text-slate-100">{hoveredBand.pulse_width > 0 ? `${hoveredBand.pulse_width.toFixed(2)} µs` : '0.00'}</span>
            </div>
            <div className="text-slate-300 font-mono">
              AoA: <span className="font-bold text-slate-100">{hoveredBand.angle_of_arrival !== null ? `${hoveredBand.angle_of_arrival}°` : 'N/A'}</span>
            </div>
            <div className="text-slate-300 font-mono">
              Hit Rate: <span className="font-bold text-rf-green-light">{hoveredBand.hit_rate}%</span> ({hoveredBand.hits}/{hoveredBand.scans})
            </div>
          </div>

          <button 
            onClick={() => onSelectBand(hoveredBand)}
            className="flex items-center gap-1 text-[11px] font-mono text-rf-cyan hover:text-rf-cyan-light hover:underline font-semibold"
          >
            <span>Inspect Band Details</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
};
