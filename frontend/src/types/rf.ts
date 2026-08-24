export type DetectionResult = 'HIT' | 'MISS';
export type DetectorState = 'ACTIVE' | 'INACTIVE';
export type PriorityLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'VERY_HIGH';

export interface Observation {
  time_slot: number;
  frequency_band: number;
  signal_power: number;
  pulse_width: number;
  angle_of_arrival: number | null;
  ground_truth_active?: boolean;
}

export interface CurrentScan {
  time_slot: number;
  frequency_band: number;
  signal_power: number;
  pulse_width: number;
  angle_of_arrival: number | null;
  detector_prediction: boolean;
  detector_state: DetectorState;
  result: DetectionResult;
  reward: number;
}

export interface BandState {
  band: number;
  signal_power: number;
  detector_state: DetectorState;
  is_selected: boolean;
  is_recently_scanned: boolean;
  estimated_hit_probability: number;
  pulse_width: number;
  angle_of_arrival: number | null;
  scans: number;
  hits: number;
  hit_rate: number;
  last_scanned: number;
  priority: PriorityLevel;
}

export interface ScanHistoryRecord {
  id: string;
  time_slot: number;
  frequency_band: number;
  signal_power: number;
  pulse_width: number;
  angle_of_arrival: number | null;
  detector_state: DetectorState;
  result: DetectionResult;
  reward: number;
}
