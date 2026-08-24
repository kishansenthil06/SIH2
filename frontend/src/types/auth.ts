export interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: string;
  callsign: string;
  clearance: 'TOP SECRET // EW-SIM' | 'SECRET // SIM-OPS' | 'TACTICAL // ANALYST';
  avatarInitials: string;
  assignedStation: string;
}

export const SAMPLE_USERS: UserProfile[] = [
  {
    id: 'usr-1',
    name: 'Major Sarah Chen',
    email: 'sarah.chen@ew.defense.sim',
    role: 'EW Operations Commander',
    callsign: 'VANGUARD-1',
    clearance: 'TOP SECRET // EW-SIM',
    avatarInitials: 'SC',
    assignedStation: 'Command Tactical Hub Alpha',
  },
  {
    id: 'usr-2',
    name: 'Dr. Avinash Sharma',
    email: 'avinash.sharma@ew.defense.sim',
    role: 'Cognitive Radar & ML Lead',
    callsign: 'SPECTRE-9',
    clearance: 'TOP SECRET // EW-SIM',
    avatarInitials: 'AS',
    assignedStation: 'Adaptive Policy Lab 4',
  },
  {
    id: 'usr-3',
    name: 'Lt. Marcus Vance',
    email: 'marcus.vance@ew.defense.sim',
    role: 'SIGINT Operations Analyst',
    callsign: 'RAVEN-4',
    clearance: 'SECRET // SIM-OPS',
    avatarInitials: 'MV',
    assignedStation: 'Electronic Surveillance Desk 2',
  },
];
