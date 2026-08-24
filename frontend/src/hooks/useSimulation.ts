import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../services/api';
import type { BandState, CurrentScan, ScanHistoryRecord } from '../types/rf';
import type { DecisionTimelineEntry, SchedulerDecision } from '../types/scheduler';
import type { PerformanceComparisonData, RewardDataPoint } from '../types/performance';
import type { EventLogItem, SimulationConfig } from '../types/simulation';
import { DEMO_SCRIPT_STEPS } from '../services/demoScript';


export function useSimulation() {
  const [isRunning, setIsRunning] = useState(false);
  const [config, setConfig] = useState<SimulationConfig>({
    speed: 2, // 500ms intervals
    total_steps: 1000,
    strategy: 'smart_ml',
    epsilon: 0.20,
    power_threshold: 5.0,
    demo_mode: false,
  });

  const [demoStepIndex, setDemoStepIndex] = useState(0);
  const [demoNarrative, setDemoNarrative] = useState<string>('');

  const [currentScan, setCurrentScan] = useState<CurrentScan>({
    time_slot: 124,
    frequency_band: 7,
    signal_power: 8.72,
    pulse_width: 1.43,
    angle_of_arrival: 42.1,
    detector_prediction: true,
    detector_state: 'ACTIVE',
    result: 'HIT',
    reward: 1.0,
  });

  const [schedulerDecision, setSchedulerDecision] = useState<SchedulerDecision>({
    next_band: 7,
    estimated_hit_probability: 0.82,
    exploration_probability: 0.20,
    strategy: 'EXPLOITATION',
    reasoning: 'Band 7 has a strong recent detection history and has not been scanned for several time slots.',
    band_probabilities: [
      { band: 7, probability: 0.82, recent_scans: 14, recent_hits: 11 },
      { band: 3, probability: 0.68, recent_scans: 8, recent_hits: 5 },
      { band: 12, probability: 0.44, recent_scans: 5, recent_hits: 2 },
      { band: 5, probability: 0.38, recent_scans: 6, recent_hits: 2 },
    ],
  });

  const [bands, setBands] = useState<BandState[]>([]);
  const [history, setHistory] = useState<ScanHistoryRecord[]>([]);
  const [timeline, setTimeline] = useState<DecisionTimelineEntry[]>([]);
  const [eventLogs, setEventLogs] = useState<EventLogItem[]>([]);
  const [rewardTrajectory, setRewardTrajectory] = useState<RewardDataPoint[]>([]);
  const [performance, setPerformance] = useState<PerformanceComparisonData | null>(null);
  const [selectedBandDetail, setSelectedBandDetail] = useState<BandState | null>(null);

  const timerRef = useRef<number | null>(null);

  const refreshData = useCallback(async () => {
    const [b, h, tl, el, rw, perf] = await Promise.all([
      api.getBands(),
      api.getHistory(),
      api.getTimeline(),
      api.getEventLogs(),
      api.getRewardTrajectory(),
      api.getPerformance(),
    ]);

    setBands(b);
    setHistory(h);
    setTimeline(tl);
    setEventLogs(el);
    setRewardTrajectory(rw);
    setPerformance(perf);
  }, []);

  // Initial load
  useEffect(() => {
    refreshData();
  }, [refreshData]);

  const stepSimulation = useCallback(async () => {
    const stepIdx = config.demo_mode ? demoStepIndex : undefined;
    const res = await api.step(config, stepIdx);

    setCurrentScan(res.currentScan);
    setSchedulerDecision(res.decision);
    if (res.narrative) {
      setDemoNarrative(res.narrative);
    }

    if (config.demo_mode) {
      setDemoStepIndex((prev) => (prev + 1) % DEMO_SCRIPT_STEPS.length);
    }

    await refreshData();
  }, [config, demoStepIndex, refreshData]);

  // Simulation run loop
  useEffect(() => {
    if (isRunning) {
      const intervalMs = Math.max(100, Math.floor(1000 / config.speed));
      timerRef.current = window.setInterval(() => {
        stepSimulation();
      }, intervalMs);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRunning, config.speed, stepSimulation]);

  const startSimulation = () => setIsRunning(true);
  const pauseSimulation = () => setIsRunning(false);
  
  const resetSimulation = async () => {
    setIsRunning(false);
    setDemoStepIndex(0);
    setDemoNarrative('');
    await api.reset();
    await refreshData();
    const cur = await api.getCurrentScan();
    setCurrentScan(cur);
  };

  const toggleDemoMode = (enabled: boolean) => {
    setConfig((prev) => ({ ...prev, demo_mode: enabled }));
    if (enabled) {
      setDemoStepIndex(0);
      setDemoNarrative(DEMO_SCRIPT_STEPS[0].narrative);
    }
  };

  const updateStrategy = (strategy: SimulationConfig['strategy']) => {
    setConfig((prev) => ({ ...prev, strategy }));
  };

  const updateSpeed = (speed: number) => {
    setConfig((prev) => ({ ...prev, speed }));
  };

  const updateEpsilon = (epsilon: number) => {
    setConfig((prev) => ({ ...prev, epsilon }));
  };

  return {
    isRunning,
    config,
    currentScan,
    schedulerDecision,
    bands,
    history,
    timeline,
    eventLogs,
    rewardTrajectory,
    performance,
    selectedBandDetail,
    demoNarrative,
    setSelectedBandDetail,
    startSimulation,
    pauseSimulation,
    resetSimulation,
    stepSimulation,
    toggleDemoMode,
    updateStrategy,
    updateSpeed,
    updateEpsilon,
  };
}
