import { useState } from 'react';
import { 
  ShieldCheck, 
  Lock, 
  User, 
  KeyRound, 
  ArrowRight, 
  Sparkles, 
  ChevronLeft,
  Radio,
  Fingerprint,
  CheckCircle2
} from 'lucide-react';
import { SAMPLE_USERS, type UserProfile } from '../types/auth';

interface LoginPageProps {
  initialUser?: UserProfile;
  onLoginSuccess: (user: UserProfile) => void;
  onBackToLanding: () => void;
}

export const LoginPage = ({ initialUser, onLoginSuccess, onBackToLanding }: LoginPageProps) => {
  const [selectedUser, setSelectedUser] = useState<UserProfile>(initialUser || SAMPLE_USERS[0]);
  const [password, setPassword] = useState('••••••••••••');
  const [isAuthenticating, setIsAuthenticating] = useState(false);

  const handleSelectUser = (user: UserProfile) => {
    setSelectedUser(user);
    setPassword('••••••••••••');
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsAuthenticating(true);
    setTimeout(() => {
      setIsAuthenticating(false);
      onLoginSuccess(selectedUser);
    }, 700);
  };

  return (
    <div className="min-h-screen bg-charcoal-950 text-slate-100 font-sans flex flex-col justify-between relative overflow-hidden selection:bg-rf-green-dark selection:text-rf-green-light">
      {/* Background Ambience */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-rf-green/5 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-10 right-10 w-96 h-96 bg-rf-cyan/5 rounded-full blur-3xl pointer-events-none" />

      {/* Top Header */}
      <div className="p-6 max-w-7xl mx-auto w-full flex items-center justify-between z-10">
        <button
          onClick={onBackToLanding}
          className="flex items-center gap-2 text-xs font-mono text-slate-400 hover:text-slate-200 transition-colors p-2 rounded-lg hover:bg-charcoal-850"
        >
          <ChevronLeft className="w-4 h-4" />
          <span>Back to Landing Page</span>
        </button>

        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-charcoal-900 border border-charcoal-800 text-[11px] font-mono text-slate-400">
          <span className="w-1.5 h-1.5 rounded-full bg-rf-green tactical-dot"></span>
          <span>SECURE EW TERMINAL // SIMULATION ACCESS</span>
        </div>
      </div>

      {/* Main Login Card */}
      <div className="max-w-md w-full mx-auto p-6 z-10">
        <div className="bg-charcoal-900/95 border border-charcoal-700/80 rounded-3xl p-8 shadow-2xl backdrop-blur-2xl relative">
          {/* Brand & Crest */}
          <div className="text-center space-y-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-rf-green-dark via-charcoal-800 to-charcoal-950 border border-rf-green-border flex items-center justify-center shadow-glow-green mx-auto">
              <span className="font-display font-black text-rf-green text-lg tracking-wider">SS</span>
            </div>
            <div>
              <h1 className="font-display font-bold text-xl text-slate-100 uppercase tracking-wide">
                OPERATOR AUTHENTICATION
              </h1>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                SMART SCAN COMMAND CENTER // CLEARANCE CHECK
              </p>
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleFormSubmit} className="space-y-4">
            {/* User Identifier */}
            <div className="space-y-1">
              <label className="text-[10px] font-mono uppercase text-slate-400 font-semibold flex items-center justify-between">
                <span>Selected Operator</span>
                <span className="text-rf-green-light">{selectedUser.callsign}</span>
              </label>
              <div className="relative">
                <User className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  readOnly
                  value={`${selectedUser.name} (${selectedUser.role})`}
                  className="w-full pl-9 pr-3 py-2.5 bg-charcoal-850 border border-charcoal-700 rounded-xl text-xs font-mono text-slate-200 focus:outline-none focus:border-rf-green"
                />
              </div>
            </div>

            {/* Email Address */}
            <div className="space-y-1">
              <label className="text-[10px] font-mono uppercase text-slate-400 font-semibold">
                Official Identifier / Email
              </label>
              <div className="relative">
                <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="email"
                  readOnly
                  value={selectedUser.email}
                  className="w-full pl-9 pr-3 py-2.5 bg-charcoal-850 border border-charcoal-700 rounded-xl text-xs font-mono text-slate-300 focus:outline-none"
                />
              </div>
            </div>

            {/* Password */}
            <div className="space-y-1">
              <label className="text-[10px] font-mono uppercase text-slate-400 font-semibold flex items-center justify-between">
                <span>Access Passcode</span>
                <span className="text-slate-500">Auto-filled for Demo</span>
              </label>
              <div className="relative">
                <KeyRound className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-9 pr-3 py-2.5 bg-charcoal-850 border border-charcoal-700 rounded-xl text-xs font-mono text-slate-200 focus:outline-none focus:border-rf-green tracking-widest"
                />
              </div>
            </div>

            {/* Clearance Level Tag */}
            <div className="p-3 rounded-xl bg-charcoal-850/80 border border-charcoal-750 flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Clearance Level:</span>
              <span className="font-bold text-rf-green-light flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5" />
                {selectedUser.clearance}
              </span>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isAuthenticating}
              className="w-full py-3 rounded-xl bg-rf-green hover:bg-rf-green-light text-charcoal-950 font-mono font-bold text-xs shadow-glow-green flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {isAuthenticating ? (
                <>
                  <Fingerprint className="w-4 h-4 animate-pulse" />
                  <span>AUTHENTICATING OPERATOR...</span>
                </>
              ) : (
                <>
                  <Radio className="w-4 h-4" />
                  <span>INITIALIZE COMMAND CENTER</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          {/* Quick Demo User Switcher */}
          <div className="mt-6 pt-5 border-t border-charcoal-800">
            <div className="text-[10px] font-mono uppercase text-slate-400 font-semibold mb-2.5 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-rf-amber" />
              <span>Select 1-Click Demo Operator</span>
            </div>

            <div className="space-y-1.5">
              {SAMPLE_USERS.map((u) => {
                const isCurrent = u.id === selectedUser.id;
                return (
                  <button
                    key={u.id}
                    type="button"
                    onClick={() => handleSelectUser(u)}
                    className={`w-full p-2 rounded-xl text-left flex items-center justify-between text-xs font-mono transition-all ${
                      isCurrent
                        ? 'bg-charcoal-800 text-rf-green-light border border-rf-green/40 shadow-sm'
                        : 'bg-charcoal-850/60 hover:bg-charcoal-800 text-slate-400 border border-charcoal-800'
                    }`}
                  >
                    <div className="flex items-center space-x-2.5">
                      <div className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-[11px] ${
                        isCurrent ? 'bg-rf-green text-charcoal-950' : 'bg-charcoal-800 text-slate-300'
                      }`}>
                        {u.avatarInitials}
                      </div>
                      <div>
                        <div className="font-bold text-slate-200">{u.name}</div>
                        <div className="text-[10px] text-slate-400">{u.role}</div>
                      </div>
                    </div>

                    {isCurrent && <CheckCircle2 className="w-4 h-4 text-rf-green shrink-0" />}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Footer */}
      <div className="p-6 text-center text-xs font-mono text-slate-500 z-10">
        Cognitive Electronic Warfare Scan Strategy Prototype // Authentication Simulation
      </div>
    </div>
  );
};
