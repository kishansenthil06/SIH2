export interface StrategyMetrics {
  detection_rate: number;
  false_alarm_rate: number;
  average_intercept_time: number;
  total_hits: number;
  total_misses: number;
  average_reward: number;
  average_scans_to_detection: number;
}

export interface PerformanceComparisonData {
  baseline: StrategyMetrics;
  ml_scheduler: StrategyMetrics;
  improvement_pct: {
    detection_rate: number;
    intercept_time: number;
    reward: number;
  };
}

export interface RewardDataPoint {
  step: number;
  baseline_reward: number;
  ml_reward: number;
  baseline_cumulative: number;
  ml_cumulative: number;
}

export interface BandEvolutionPoint {
  step: number;
  [key: string]: number; // band probabilities or hit rates
}
