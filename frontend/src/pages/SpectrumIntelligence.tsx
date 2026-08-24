import React, { useState } from 'react';
import type { BandState } from '../types/rf';
import { 
  Activity, 
  Sliders, 
  Search
} from 'lucide-react';

import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

interface SpectrumIntelligenceProps {
  bands: BandState[];
  timeSlot: number;
  onSelectBand: (band: BandState) => void;
}

type ViewMetric = 'power' | 'activity' | 'priority' | 'hit_rate';

export const SpectrumIntelligence: React.FC<SpectrumIntelligenceProps> = ({
  bands,
  timeSlot,
  onSelectBand,
}) => {
  const [activeMetric, setActiveMetric] = useState<ViewMetric>('power');
  const [selectedSlot, setSelectedSlot] = useState<number>(timeSlot);
  const [searchFilter, setSearchFilter] = useState('');

  // Prepare chart data based on selected view mode
  const chartData = bands.map((b) => {
    let value = 0;
    if (activeMetric === 'power') value = Math.max(0, b.signal_power + 5);
    else if (activeMetric === 'activity') value = b.hits;
    else if (activeMetric === 'priority') {
      value = { LOW: 25, MEDIUM: 50, HIGH: 75, VERY_HIGH: 100 }[b.priority];
    } else if (activeMetric === 'hit_rate') value = b.hit_rate;

    return {
      band: `F${b.band}`,
      rawBand: b.band,
      value,
      realPower: b.signal_power,
      hits: b.hits,
      scans: b.scans,
      hitRate: b.hit_rate,
      priority: b.priority,
      state: b.detector_state,
    };
  });

  const filteredBands = bands.filter((b) => 
    `band ${b.band}`.toLowerCase().includes(searchFilter.toLowerCase()) ||
    b.priority.toLowerCase().includes(searchFilter.toLowerCase())
  );

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-charcoal-750/80">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-rf-cyan-bg border border-rf-cyan-border text-rf-cyan">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-display font-bold text-lg text-slate-100 uppercase tracking-wide">
                RF SPECTRUM INTELLIGENCE
              </h2>
              <p className="text-xs text-slate-400 font-sans">
                Full-Band Spectral Distribution & Temporal Observation Analysis
              </p>
            </div>
          </div>
        </div>

        {/* View Metric Mode Switcher */}
        <div className="flex items-center space-x-1.5 p-1 rounded-lg bg-charcoal-850 border border-charcoal-750 text-xs font-mono">
          <button
            onClick={() => setActiveMetric('power')}
            className={`px-3 py-1.5 rounded-md font-semibold transition-all ${
              activeMetric === 'power'
                ? 'bg-rf-green text-charcoal-950 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Signal Power (dB)
          </button>
          <button
            onClick={() => setActiveMetric('hit_rate')}
            className={`px-3 py-1.5 rounded-md font-semibold transition-all ${
              activeMetric === 'hit_rate'
                ? 'bg-rf-cyan text-charcoal-950 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Hit Rate (%)
          </button>
          <button
            onClick={() => setActiveMetric('activity')}
            className={`px-3 py-1.5 rounded-md font-semibold transition-all ${
              activeMetric === 'activity'
                ? 'bg-rf-amber text-charcoal-950 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Cumulative Hits
          </button>
          <button
            onClick={() => setActiveMetric('priority')}
            className={`px-3 py-1.5 rounded-md font-semibold transition-all ${
              activeMetric === 'priority'
                ? 'bg-slate-200 text-charcoal-950 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            ML Priority
          </button>
        </div>
      </div>

      {/* Main Interactive Spectrum Intelligence Chart */}
      <div className="p-6 rounded-xl bg-charcoal-900 border border-charcoal-750/80 space-y-4">
        <div className="flex items-center justify-between">
          <div className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider">
            20-Band Frequency Domain Analysis
          </div>
          <div className="text-xs font-mono text-slate-400">
            Viewing: <strong className="text-slate-200 uppercase">{activeMetric.replace('_', ' ')}</strong>
          </div>
        </div>

        {/* Recharts Bar Display */}
        <div className="h-72 w-full bg-charcoal-950/90 rounded-lg p-3 border border-charcoal-800">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <XAxis dataKey="band" stroke="#64748b" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="p-3 bg-charcoal-900 border border-charcoal-700 rounded-lg shadow-xl text-xs font-mono">
                        <div className="font-bold text-slate-100 mb-1">BAND {d.rawBand}</div>
                        <div className="text-slate-300">Power: <strong className="text-rf-green-light">{d.realPower.toFixed(2)} dB</strong></div>
                        <div className="text-slate-300">Hit Rate: <strong className="text-rf-cyan-light">{d.hitRate}%</strong></div>
                        <div className="text-slate-300">Hits: <strong>{d.hits} / {d.scans}</strong></div>
                        <div className="text-slate-300">Priority: <strong className="text-rf-amber-light">{d.priority}</strong></div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {chartData.map((entry, index) => {
                  let fillColor = '#33445c';
                  if (activeMetric === 'power') {
                    fillColor = entry.realPower > 5 ? '#10b981' : '#33445c';
                  } else if (activeMetric === 'hit_rate') {
                    fillColor = entry.hitRate > 50 ? '#06b6d4' : entry.hitRate > 20 ? '#0284c7' : '#1e293b';
                  } else if (activeMetric === 'activity') {
                    fillColor = entry.hits > 10 ? '#f59e0b' : '#33445c';
                  } else if (activeMetric === 'priority') {
                    fillColor = entry.priority === 'VERY_HIGH' ? '#f59e0b' : entry.priority === 'HIGH' ? '#10b981' : '#33445c';
                  }
                  return <Cell key={`cell-${index}`} fill={fillColor} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Time Slot Scrubber */}
        <div className="pt-2 p-4 rounded-lg bg-charcoal-850/80 border border-charcoal-750">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-rf-cyan" />
              <span className="text-xs font-mono font-bold text-slate-200">
                HISTORICAL TIME SLOT SCRUBBER
              </span>
            </div>
            <span className="text-xs font-mono text-rf-green-light font-bold">
              Time Slot #{selectedSlot} / #5000
            </span>
          </div>

          <input
            type="range"
            min={0}
            max={Math.max(500, timeSlot + 10)}
            value={selectedSlot}
            onChange={(e) => setSelectedSlot(Number(e.target.value))}
            className="w-full h-1.5 bg-charcoal-950 rounded-lg appearance-none cursor-pointer accent-rf-green"
          />
          <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
            <span>T=0</span>
            <span>T=1250</span>
            <span>T=2500</span>
            <span>T=3750</span>
            <span>T=5000</span>
          </div>
        </div>
      </div>

      {/* Band Intelligence Table */}
      <div className="p-5 rounded-xl bg-charcoal-900 border border-charcoal-750/80">
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-charcoal-750/80">
          <div>
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              BAND INTELLIGENCE & RECEPTIVITY MATRIX
            </h3>
            <p className="text-[11px] text-slate-400 font-sans">
              Continuous Empirical Statistics per Frequency Channel
            </p>
          </div>

          {/* Search Filter */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Filter bands or priority..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              className="pl-8 pr-3 py-1.5 bg-charcoal-850 border border-charcoal-700 rounded-lg text-xs font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:border-rf-green"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-charcoal-750/80 text-[10px] text-slate-400 uppercase tracking-wider bg-charcoal-850/50">
                <th className="py-2.5 px-3">Band</th>
                <th className="py-2.5 px-3">Frequency Range</th>
                <th className="py-2.5 px-3">Dwells / Scans</th>
                <th className="py-2.5 px-3">Hits Detected</th>
                <th className="py-2.5 px-3">Empirical Hit Rate</th>
                <th className="py-2.5 px-3">Current Power</th>
                <th className="py-2.5 px-3">Last Scanned Slot</th>
                <th className="py-2.5 px-3 text-right">ML Priority</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-charcoal-800/60">
              {filteredBands.map((b) => (
                <tr
                  key={b.band}
                  onClick={() => onSelectBand(b)}
                  className="hover:bg-charcoal-850/60 cursor-pointer transition-colors"
                >
                  <td className="py-2.5 px-3 font-bold text-slate-100">
                    <span className="px-2 py-0.5 rounded bg-charcoal-800 border border-charcoal-700">
                      BAND {b.band}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">
                    {(2.0 + (b.band - 1) * 0.8).toFixed(1)} – {(2.8 + (b.band - 1) * 0.8).toFixed(1)} GHz
                  </td>
                  <td className="py-2.5 px-3 text-slate-300 font-semibold">{b.scans}</td>
                  <td className="py-2.5 px-3 text-rf-green-light font-bold">{b.hits}</td>
                  <td className="py-2.5 px-3">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-slate-200">{b.hit_rate}%</span>
                      <div className="w-16 h-1.5 bg-charcoal-950 rounded-full overflow-hidden">
                        <div
                          style={{ width: `${b.hit_rate}%` }}
                          className="h-full bg-rf-green rounded-full"
                        />
                      </div>
                    </div>
                  </td>
                  <td className={`py-2.5 px-3 font-bold ${b.signal_power > 5 ? 'text-rf-green-light' : 'text-slate-400'}`}>
                    {b.signal_power.toFixed(2)} dB
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">
                    {b.last_scanned > 0 ? `#${b.last_scanned}` : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold ${
                        b.priority === 'VERY_HIGH'
                          ? 'bg-rf-amber-bg text-rf-amber-light border border-rf-amber-border'
                          : b.priority === 'HIGH'
                          ? 'bg-rf-green-bg text-rf-green-light border border-rf-green-border'
                          : b.priority === 'MEDIUM'
                          ? 'bg-rf-cyan-bg text-rf-cyan-light border border-rf-cyan-border'
                          : 'bg-charcoal-800 text-slate-400'
                      }`}
                    >
                      {b.priority}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
