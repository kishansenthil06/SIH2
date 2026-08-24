export type SchedulerStrategy = 'EXPLOITATION' | 'EXPLORATION';

export interface BandProbability {
  band: number;
  probability: number;
  recent_scans: number;
  recent_hits: number;
}

export interface SchedulerDecision {
  next_band: number;
  estimated_hit_probability: number;
  exploration_probability: number;
  strategy: SchedulerStrategy;
  reasoning: string;
  band_probabilities: BandProbability[];
}

export interface DecisionTimelineEntry {
  step: number;
  time_slot: number;
  selected_band: number;
  strategy: SchedulerStrategy;
  hit_probability: number;
  result: 'HIT' | 'MISS';
  reward: number;
}
