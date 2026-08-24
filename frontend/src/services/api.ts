import { mockEngine } from './mockApi';
import { DEMO_SCRIPT_STEPS } from './demoScript';
import type { BandState, CurrentScan, ScanHistoryRecord } from '../types/rf';
import type { DecisionTimelineEntry, SchedulerDecision } from '../types/scheduler';
import type { PerformanceComparisonData, RewardDataPoint } from '../types/performance';
import type { EventLogItem, SimulationConfig } from '../types/simulation';


const API_BASE_URL = 'http://127.0.0.1:8080';

class ApiService {
  private isBackendOnline = false;

  constructor() {
    this.checkHealth();
  }

  public async checkHealth(): Promise<boolean> {
    try {
      const res = await fetch(`${API_BASE_URL}/api/status`, { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        this.isBackendOnline = true;
        return true;
      }
    } catch {
      this.isBackendOnline = false;
    }
    return false;
  }

  public getBackendStatus(): boolean {
    return this.isBackendOnline;
  }

  public async getStatus() {
    if (this.isBackendOnline) {
      try {
        const res = await fetch(`${API_BASE_URL}/api/status`);
        return await res.json();
      } catch {
        this.isBackendOnline = false;
      }
    }
    return {
      status: 'online',
      mode: 'Simulation / Demo Engine',
      dataset: 'Temporary RF Dataset (100,000 observations)',
      bands: 20,
    };
  }

  public async getBands(): Promise<BandState[]> {
    return mockEngine.getBands();
  }

  public async getCurrentScan(): Promise<CurrentScan> {
    return mockEngine.getCurrentScan();
  }

  public async getSchedulerDecision(config: SimulationConfig): Promise<SchedulerDecision> {
    return mockEngine.getSchedulerDecision(config);
  }

  public async getHistory(): Promise<ScanHistoryRecord[]> {
    return mockEngine.getHistory();
  }

  public async getEventLogs(): Promise<EventLogItem[]> {
    return mockEngine.getEventLogs();
  }

  public async getTimeline(): Promise<DecisionTimelineEntry[]> {
    return mockEngine.getTimeline();
  }

  public async getRewardTrajectory(): Promise<RewardDataPoint[]> {
    return mockEngine.getRewardTrajectory();
  }

  public async getPerformance(): Promise<PerformanceComparisonData> {
    return mockEngine.getPerformanceComparison();
  }

  public async step(config: SimulationConfig, demoStepIndex?: number): Promise<{ currentScan: CurrentScan; decision: SchedulerDecision; narrative?: string }> {
    if (config.demo_mode && demoStepIndex !== undefined) {
      const scriptStep = DEMO_SCRIPT_STEPS[demoStepIndex % DEMO_SCRIPT_STEPS.length];
      return {
        currentScan: scriptStep.scan,
        decision: scriptStep.decision,
        narrative: scriptStep.narrative,
      };
    }

    return mockEngine.step(config);
  }

  public async reset() {
    mockEngine.reset();
  }
}

export const api = new ApiService();
