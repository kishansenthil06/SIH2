import React from 'react';
import { 
  Radio, 
  Target, 
  CheckCircle2, 
  Award, 
  Cpu, 
  Sparkles 
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { ArchitectureFlow } from '../components/ArchitectureFlow';
import { SpectrumChart } from '../components/SpectrumChart';
import { CurrentScan } from '../components/CurrentScan';
import { MLDecision } from '../components/MLDecision';
import { ScanHistory } from '../components/ScanHistory';
import type { BandState, CurrentScan as CurrentScanType, ScanHistoryRecord } from '../types/rf';
import type { SchedulerDecision } from '../types/scheduler';


interface CommandCenterProps {
  isRunning: boolean;
  demoMode: boolean;
  demoNarrative: string;
  currentScan: CurrentScanType;
  schedulerDecision: SchedulerDecision;
  bands: BandState[];
  history: ScanHistoryRecord[];
  onSelectBand: (band: BandState) => void;
}

export const CommandCenter: React.FC<CommandCenterProps> = ({
  isRunning,
  demoMode,
  demoNarrative,
  currentScan,
  schedulerDecision,
  bands,
  history,
  onSelectBand,
}) => {
  const isHit = currentScan.result === 'HIT';

  return (
    <div className="space-y-6 pb-12">
      {/* Demo Mode Live Presentation Story Banner */}
      {demoMode && demoNarrative && (
        <div className="p-3.5 rounded-xl bg-gradient-to-r from-rf-amber/20 via-charcoal-850 to-charcoal-900 border border-rf-amber-border flex items-center gap-3 animate-in fade-in">
          <div className="p-2 rounded-lg bg-rf-amber-bg text-rf-amber shrink-0">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] font-mono uppercase font-bold text-rf-amber-light tracking-wider">
              Hackathon Live Demo Sequence
            </div>
            <div className="text-xs font-sans text-slate-100 mt-0.5">
              {demoNarrative}
            </div>
          </div>
        </div>
      )}

      {/* 1. TOP KPI ROW */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3.5">
        <MetricCard
          title="Current Time Slot"
          value={`#${currentScan.time_slot}`}
          subtitle="Simulation Clock"
          icon={Radio}
          variant="cyan"
        />

        <MetricCard
          title="Selected Band"
          value={`BAND ${currentScan.frequency_band}`}
          subtitle="Tuned RF Channel"
          icon={Target}
          variant="green"
          isGlowing
        />

        <MetricCard
          title="Detection Status"
          value={currentScan.result === 'HIT' ? 'ACTIVE / HIT' : 'INACTIVE / MISS'}
          subtitle={`Detector: ${currentScan.detector_state}`}
          icon={CheckCircle2}
          variant={isHit ? 'green' : 'amber'}
        />

        <MetricCard
          title="Current Reward"
          value={`${currentScan.reward > 0 ? '+' : ''}${currentScan.reward.toFixed(1)} J`}
          subtitle="Reinforcement Feedback"
          icon={Award}
          variant={currentScan.reward > 0 ? 'green' : 'neutral'}
        />

        <MetricCard
          title="ML Confidence"
          value={`${(schedulerDecision.estimated_hit_probability * 100).toFixed(0)}%`}
          subtitle={`Mode: ${schedulerDecision.strategy}`}
          icon={Cpu}
          variant="green"
        />
      </div>

      {/* 2. COMPACT ARCHITECTURE DATA FLOW */}
      <ArchitectureFlow
        isRunning={isRunning}
        currentBand={currentScan.frequency_band}
        detectorActive={currentScan.detector_state === 'ACTIVE'}
        result={currentScan.result}
        reward={currentScan.reward}
        strategy={schedulerDecision.strategy}
      />

      {/* 3. MAIN SPECTRUM VISUALIZATION & CURRENT SCAN */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2">
          <SpectrumChart
            bands={bands}
            selectedBand={currentScan.frequency_band}
            onSelectBand={onSelectBand}
          />
        </div>
        <div className="lg:col-span-1">
          <CurrentScan scan={currentScan} />
        </div>
      </div>

      {/* 4. ML SCHEDULER DECISION PANEL */}
      <MLDecision decision={schedulerDecision} />

      {/* 5. SCAN HISTORY */}
      <ScanHistory history={history} />
    </div>
  );
};
