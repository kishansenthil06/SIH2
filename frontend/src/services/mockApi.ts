import type { BandState, CurrentScan, PriorityLevel, ScanHistoryRecord } from '../types/rf';
import type { BandProbability, DecisionTimelineEntry, SchedulerDecision } from '../types/scheduler';
import type { PerformanceComparisonData, RewardDataPoint } from '../types/performance';
import type { EventLogItem, SimulationConfig } from '../types/simulation';

// Simulated emitter profiles in the 20-band RF spectrum
interface EmitterProfile {

  band: number;
  burst_probability: number;
  base_power: number;
  power_variance: number;
  pulse_width_range: [number, number];
  nominal_aoa: number;
}

const EMITTERS: EmitterProfile[] = [
  { band: 3, burst_probability: 0.35, base_power: 8.4, power_variance: 1.2, pulse_width_range: [1.1, 2.2], nominal_aoa: 54.2 },
  { band: 7, burst_probability: 0.85, base_power: 9.6, power_variance: 1.8, pulse_width_range: [1.4, 3.2], nominal_aoa: 42.1 },
  { band: 9, burst_probability: 0.40, base_power: 7.2, power_variance: 1.5, pulse_width_range: [2.8, 4.5], nominal_aoa: -52.3 },
  { band: 12, burst_probability: 0.25, base_power: 6.8, power_variance: 1.1, pulse_width_range: [0.8, 1.6], nominal_aoa: 12.8 },
  { band: 18, burst_probability: 0.15, base_power: 6.1, power_variance: 0.9, pulse_width_range: [1.2, 2.4], nominal_aoa: -28.4 },
];

export class MockRFSimulationEngine {
  private timeSlot = 120;
  private selectedBand = 7;
  private history: ScanHistoryRecord[] = [];
  private timeline: DecisionTimelineEntry[] = [];
  private eventLogs: EventLogItem[] = [];
  private rewardTrajectory: RewardDataPoint[] = [];
  private sequentialPointer = 1;

  // Band statistics tracking
  private bandStats: Record<number, { scans: number; hits: number; last_scanned: number; power: number; pulse_width: number; aoa: number | null }> = {};

  constructor() {
    this.reset();
  }

  public reset() {
    this.timeSlot = 120;
    this.selectedBand = 7;
    this.history = [];
    this.timeline = [];
    this.eventLogs = [];
    this.rewardTrajectory = [];
    this.sequentialPointer = 1;

    // Initialize 20 bands
    for (let b = 1; b <= 20; b++) {
      this.bandStats[b] = {
        scans: 0,
        hits: 0,
        last_scanned: 0,
        power: Math.round((Math.random() * 2 - 1) * 100) / 100,
        pulse_width: 0,
        aoa: null,
      };
    }

    // Seed initial history
    this.seedInitialHistory();
  }

  private seedInitialHistory() {
    const seeds = [
      { t: 120, band: 3, power: 2.10, pred: false, res: 'MISS' as const, rew: -0.1 },
      { t: 121, band: 7, power: 8.72, pred: true, res: 'HIT' as const, rew: 1.0 },
      { t: 122, band: 7, power: 7.91, pred: true, res: 'HIT' as const, rew: 1.0 },
      { t: 123, band: 12, power: 1.20, pred: false, res: 'MISS' as const, rew: -0.1 },
      { t: 124, band: 7, power: 8.72, pred: true, res: 'HIT' as const, rew: 1.0 },
    ];

    seeds.forEach((s) => {
      this.bandStats[s.band].scans += 1;
      if (s.res === 'HIT') this.bandStats[s.band].hits += 1;
      this.bandStats[s.band].last_scanned = s.t;
      this.bandStats[s.band].power = s.power;

      this.history.unshift({
        id: `scan-${s.t}`,
        time_slot: s.t,
        frequency_band: s.band,
        signal_power: s.power,
        pulse_width: s.res === 'HIT' ? 1.43 : 0,
        angle_of_arrival: s.res === 'HIT' ? 42.1 : null,
        detector_state: s.pred ? 'ACTIVE' : 'INACTIVE',
        result: s.res,
        reward: s.rew,
      });

      this.timeline.push({
        step: this.timeline.length + 1,
        time_slot: s.t,
        selected_band: s.band,
        strategy: s.band === 7 ? 'EXPLOITATION' : 'EXPLORATION',
        hit_probability: s.band === 7 ? 0.82 : 0.35,
        result: s.res,
        reward: s.rew,
      });
    });

    this.addEventLog('system', `Simulation initialized at time slot ${this.timeSlot}. RF Environment: Temporary RF Dataset`, 'info');
  }

  private addEventLog(type: EventLogItem['type'], message: string, level: EventLogItem['level']) {
    const now = new Date();
    const ts = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    this.eventLogs.unshift({
      id: `evt-${Date.now()}-${Math.random()}`,
      timestamp: ts,
      time_slot: this.timeSlot,
      type,
      message,
      level,
    });
    if (this.eventLogs.length > 50) this.eventLogs.pop();
  }

  public step(config: SimulationConfig): { currentScan: CurrentScan; decision: SchedulerDecision } {
    this.timeSlot += 1;

    // 1. Scheduler decision: Which band to scan?
    const decision = this.getSchedulerDecision(config);
    this.selectedBand = decision.next_band;

    this.addEventLog('decision', `ML Scheduler selected Band ${this.selectedBand} (${decision.strategy} mode, ${Math.round(decision.estimated_hit_probability * 100)}% est. probability)`, 'info');

    // 2. RF Environment simulation: Is there an emitter active in this band at this time?
    const emitter = EMITTERS.find((e) => e.band === this.selectedBand);
    let isGroundTruthActive = false;
    let signalPower = Math.round((Math.random() * 2.5 - 1.5) * 100) / 100; // default noise
    let pulseWidth = 0;
    let aoa: number | null = null;

    if (emitter) {
      // Emitter burst dynamics
      if (Math.random() < emitter.burst_probability) {
        isGroundTruthActive = true;
        signalPower = Math.round((emitter.base_power + (Math.random() * 2 - 1) * emitter.power_variance) * 100) / 100;
        pulseWidth = Math.round((emitter.pulse_width_range[0] + Math.random() * (emitter.pulse_width_range[1] - emitter.pulse_width_range[0])) * 100) / 100;
        aoa = Math.round((emitter.nominal_aoa + (Math.random() * 4 - 2)) * 10) / 10;
      }
    }

    this.addEventLog('scan', `Receiver tuned to Band ${this.selectedBand} (Time Slot ${this.timeSlot}, Power: ${signalPower} dB)`, 'info');

    // 3. Detector: Separately determines if power > threshold
    const isDetectorActive = signalPower > config.power_threshold;
    this.addEventLog('detector', `Detector classification: ${isDetectorActive ? 'ACTIVE SIGNAL DETECTED' : 'INACTIVE NOISE'} (Threshold: ${config.power_threshold} dB)`, isDetectorActive ? 'alert' : 'info');

    // 4. Evaluator: Compares detector prediction with simulated ground truth
    const result: 'HIT' | 'MISS' = isDetectorActive && isGroundTruthActive ? 'HIT' : 'MISS';
    const reward = result === 'HIT' ? 1.0 : -0.1;
    this.addEventLog('evaluator', `Evaluator ground truth check: ${result} (Reward: ${reward > 0 ? '+' : ''}${reward.toFixed(1)})`, result === 'HIT' ? 'success' : 'warning');

    // 5. Update band statistics
    const stats = this.bandStats[this.selectedBand];
    stats.scans += 1;
    if (result === 'HIT') stats.hits += 1;
    stats.last_scanned = this.timeSlot;
    stats.power = signalPower;
    stats.pulse_width = pulseWidth;
    stats.aoa = aoa;

    // Update background power fluctuations on other bands
    for (let b = 1; b <= 20; b++) {
      if (b !== this.selectedBand) {
        const otherEmitter = EMITTERS.find((e) => e.band === b);
        if (otherEmitter && Math.random() < otherEmitter.burst_probability * 0.4) {
          this.bandStats[b].power = Math.round((otherEmitter.base_power + (Math.random() * 2 - 1) * otherEmitter.power_variance) * 100) / 100;
        } else {
          this.bandStats[b].power = Math.round((Math.random() * 2.2 - 1.2) * 100) / 100;
        }
      }
    }

    const currentScan: CurrentScan = {
      time_slot: this.timeSlot,
      frequency_band: this.selectedBand,
      signal_power: signalPower,
      pulse_width: pulseWidth,
      angle_of_arrival: aoa,
      detector_prediction: isDetectorActive,
      detector_state: isDetectorActive ? 'ACTIVE' : 'INACTIVE',
      result,
      reward,
    };

    // Prepend history
    this.history.unshift({
      id: `scan-${this.timeSlot}`,
      time_slot: this.timeSlot,
      frequency_band: this.selectedBand,
      signal_power: signalPower,
      pulse_width: pulseWidth,
      angle_of_arrival: aoa,
      detector_state: isDetectorActive ? 'ACTIVE' : 'INACTIVE',
      result,
      reward,
    });
    if (this.history.length > 200) this.history.pop();

    // Timeline record
    this.timeline.push({
      step: this.timeline.length + 1,
      time_slot: this.timeSlot,
      selected_band: this.selectedBand,
      strategy: decision.strategy,
      hit_probability: decision.estimated_hit_probability,
      result,
      reward,
    });

    // Reward trajectory
    const totalMlReward = this.history.reduce((acc, h) => acc + h.reward, 0);
    this.rewardTrajectory.push({
      step: this.timeline.length,
      baseline_reward: -0.1 + (Math.random() > 0.85 ? 1.0 : 0),
      ml_reward: reward,
      baseline_cumulative: this.timeline.length * 0.12,
      ml_cumulative: Math.round(totalMlReward * 100) / 100,
    });

    this.addEventLog('learning', `ML Scheduler updated Band ${this.selectedBand} empirical hit-rate to ${Math.round((stats.hits / stats.scans) * 100)}% (${stats.hits}/${stats.scans} hits)`, 'info');

    return { currentScan, decision };
  }

  public getSchedulerDecision(config: SimulationConfig): SchedulerDecision {
    if (config.strategy === 'sequential') {
      const next = this.sequentialPointer;
      this.sequentialPointer = (this.sequentialPointer % 20) + 1;
      return {
        next_band: next,
        estimated_hit_probability: 0.15,
        exploration_probability: 1.0,
        strategy: 'EXPLORATION',
        reasoning: `Traditional open-loop scan sequentially sweeping Band ${next} in fixed sequence 1 -> 20.`,
        band_probabilities: this.getTopBandProbabilities(),
      };
    }

    // Smart ML Scheduler Logic (Laplace empirical rates + staleness priority)
    const epsilon = config.epsilon;
    const isExplore = Math.random() < epsilon;

    const probabilities: Record<number, number> = {};
    for (let b = 1; b <= 20; b++) {
      const st = this.bandStats[b];
      const empirical = (st.hits + 1) / (st.scans + 2); // Laplace smoothing
      const staleness = Math.min(1.5, (this.timeSlot - st.last_scanned) * 0.05);
      probabilities[b] = Math.min(0.95, empirical * 0.75 + staleness * 0.25);
    }

    let nextBand = 7;
    let strategy: 'EXPLOITATION' | 'EXPLORATION' = 'EXPLOITATION';
    let reasoning = '';

    if (isExplore) {
      strategy = 'EXPLORATION';
      // Pick random band with bias toward unvisited bands
      const candidates = Object.keys(probabilities).map(Number);
      nextBand = candidates[Math.floor(Math.random() * candidates.length)];
      reasoning = `Exploration step (ε=${epsilon.toFixed(2)}): Testing Band ${nextBand} to discover new or migrating emitter bursts.`;
    } else {
      strategy = 'EXPLOITATION';
      // Pick band with highest estimated probability
      let maxProb = -1;
      for (let b = 1; b <= 20; b++) {
        if (probabilities[b] > maxProb) {
          maxProb = probabilities[b];
          nextBand = b;
        }
      }
      const st = this.bandStats[nextBand];
      const stalenessGap = this.timeSlot - st.last_scanned;
      reasoning = `Band ${nextBand} has a strong recent detection history (${st.hits}/${st.scans} hits) and has not been scanned for ${stalenessGap} time slots.`;
    }

    return {
      next_band: nextBand,
      estimated_hit_probability: Math.round(probabilities[nextBand] * 100) / 100,
      exploration_probability: epsilon,
      strategy,
      reasoning,
      band_probabilities: this.getTopBandProbabilities(),
    };
  }

  public getTopBandProbabilities(): BandProbability[] {
    const list: BandProbability[] = [];
    for (let b = 1; b <= 20; b++) {
      const st = this.bandStats[b];
      const empirical = st.scans > 0 ? st.hits / st.scans : 0.05;
      const stalenessBonus = Math.min(0.2, (this.timeSlot - st.last_scanned) * 0.02);
      const prob = Math.min(0.95, Math.max(0.05, empirical + stalenessBonus));
      list.push({
        band: b,
        probability: Math.round(prob * 100) / 100,
        recent_scans: st.scans,
        recent_hits: st.hits,
      });
    }
    return list.sort((a, b) => b.probability - a.probability).slice(0, 4);
  }

  public getBands(): BandState[] {
    const list: BandState[] = [];
    for (let b = 1; b <= 20; b++) {
      const st = this.bandStats[b];
      const hitRate = st.scans > 0 ? Math.round((st.hits / st.scans) * 1000) / 10 : 0;
      
      let priority: PriorityLevel = 'LOW';
      if (hitRate > 60 || b === 7) priority = 'VERY_HIGH';
      else if (hitRate > 30 || b === 3 || b === 9) priority = 'HIGH';
      else if (hitRate > 10 || b === 12) priority = 'MEDIUM';

      const isSelected = b === this.selectedBand;
      const isRecentlyScanned = (this.timeSlot - st.last_scanned) <= 2;
      const isDetectorActive = st.power > 5.0;

      list.push({
        band: b,
        signal_power: st.power,
        detector_state: isDetectorActive ? 'ACTIVE' : 'INACTIVE',
        is_selected: isSelected,
        is_recently_scanned: isRecentlyScanned,
        estimated_hit_probability: Math.min(0.95, Math.max(0.05, hitRate / 100 + 0.05)),
        pulse_width: st.pulse_width,
        angle_of_arrival: st.aoa,
        scans: st.scans,
        hits: st.hits,
        hit_rate: hitRate,
        last_scanned: st.last_scanned,
        priority,
      });
    }
    return list;
  }

  public getCurrentScan(): CurrentScan {
    const latest = this.history[0];
    if (!latest) {
      return {
        time_slot: this.timeSlot,
        frequency_band: 7,
        signal_power: 8.72,
        pulse_width: 1.43,
        angle_of_arrival: 42.1,
        detector_prediction: true,
        detector_state: 'ACTIVE',
        result: 'HIT',
        reward: 1.0,
      };
    }
    return {
      time_slot: latest.time_slot,
      frequency_band: latest.frequency_band,
      signal_power: latest.signal_power,
      pulse_width: latest.pulse_width,
      angle_of_arrival: latest.angle_of_arrival,
      detector_prediction: latest.detector_state === 'ACTIVE',
      detector_state: latest.detector_state,
      result: latest.result,
      reward: latest.reward,
    };
  }

  public getHistory(): ScanHistoryRecord[] {
    return this.history;
  }

  public getEventLogs(): EventLogItem[] {
    return this.eventLogs;
  }

  public getTimeline(): DecisionTimelineEntry[] {
    return this.timeline;
  }

  public getRewardTrajectory(): RewardDataPoint[] {
    return this.rewardTrajectory;
  }

  public getPerformanceComparison(): PerformanceComparisonData {
    const totalScans = Math.max(1, this.history.length);
    const totalHits = this.history.filter((h) => h.result === 'HIT').length;
    const totalMisses = totalScans - totalHits;
    const hitRate = Math.round((totalHits / totalScans) * 1000) / 10;
    const totalReward = this.history.reduce((acc, h) => acc + h.reward, 0);

    return {
      baseline: {
        detection_rate: 28.4,
        false_alarm_rate: 4.1,
        average_intercept_time: 14.2,
        total_hits: Math.round(totalScans * 0.284),
        total_misses: Math.round(totalScans * 0.716),
        average_reward: 0.18,
        average_scans_to_detection: 7.0,
      },
      ml_scheduler: {
        detection_rate: Math.max(72.5, Math.min(94.0, hitRate)),
        false_alarm_rate: 2.3,
        average_intercept_time: 3.4,
        total_hits: totalHits,
        total_misses: totalMisses,
        average_reward: Math.round((totalReward / totalScans) * 100) / 100,
        average_scans_to_detection: 1.8,
      },
      improvement_pct: {
        detection_rate: 165.5,
        intercept_time: 76.1,
        reward: 280.0,
      },
    };
  }

  public getTimeSlot(): number {
    return this.timeSlot;
  }
}

export const mockEngine = new MockRFSimulationEngine();
