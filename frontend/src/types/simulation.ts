export type ActiveStrategy = 'smart_ml' | 'sequential';

export interface SimulationConfig {
  speed: number; // 1 = 1s, 2 = 500ms, 5 = 200ms, 10 = 100ms
  total_steps: number;
  strategy: ActiveStrategy;
  epsilon: number; // exploration rate e.g. 0.20
  power_threshold: number; // default 5.0 dB
  demo_mode: boolean;
}

export interface EventLogItem {
  id: string;
  timestamp: string;
  time_slot: number;
  type: 'decision' | 'scan' | 'detector' | 'evaluator' | 'learning' | 'system';
  message: string;
  level: 'info' | 'success' | 'warning' | 'alert';
}
