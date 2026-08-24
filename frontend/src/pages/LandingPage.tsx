import { 
  Radio, 
  Cpu, 
  ShieldCheck, 
  ArrowRight, 
  Lock, 
  Sparkles, 
  CheckCircle2, 
  Terminal,
  Database
} from 'lucide-react';

import type { UserProfile } from '../types/auth';
import { SAMPLE_USERS } from '../types/auth';

interface LandingPageProps {
  onGoToLogin: (selectedUser?: UserProfile) => void;
}

export const LandingPage = ({ onGoToLogin }: LandingPageProps) => {

  return (
    <div className="min-h-screen bg-charcoal-950 text-slate-100 font-sans selection:bg-rf-green-dark selection:text-rf-green-light">
      {/* 1. TOP FLOATING BRAND HEADER */}
      <header className="fixed top-0 inset-x-0 z-50 bg-charcoal-950/80 backdrop-blur-xl border-b border-charcoal-800/80 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-rf-green-dark via-charcoal-800 to-charcoal-950 border border-rf-green-border flex items-center justify-center shadow-glow-green">
              <span className="font-display font-black text-rf-green text-sm tracking-wider">SS</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-display font-bold text-slate-100 text-sm tracking-wider">
                  SMART SCAN
                </span>
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-rf-green-bg text-rf-green border border-rf-green-border">
                  EW COMMAND
                </span>
              </div>
              <span className="text-[10px] text-slate-400 font-mono">Cognitive Spectrum Intelligence</span>
            </div>
          </div>

          <div className="flex items-center space-x-3 text-xs font-mono">
            <span className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-charcoal-850 border border-charcoal-750 text-slate-400 text-[11px]">
              <span className="w-1.5 h-1.5 rounded-full bg-rf-amber tactical-dot"></span>
              AUTHENTICATION REQUIRED
            </span>
            <button
              onClick={() => onGoToLogin()}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-rf-green text-charcoal-950 font-bold shadow-glow-green hover:bg-rf-green-light transition-all"
            >
              <Lock className="w-3.5 h-3.5" />
              <span>Operator Sign In</span>
            </button>
          </div>
        </div>
      </header>

      {/* 2. HERO SECTION */}
      <section className="pt-32 pb-20 px-6 max-w-7xl mx-auto relative overflow-hidden">
        {/* Subtle decorative background glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-rf-green/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute top-1/3 right-10 w-72 h-72 bg-rf-cyan/10 rounded-full blur-3xl pointer-events-none" />

        <div className="text-center space-y-6 max-w-4xl mx-auto relative z-10">
          {/* Classification & Protocol Tag */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-charcoal-900 border border-charcoal-750 text-xs font-mono text-slate-300 shadow-xl">
            <Sparkles className="w-3.5 h-3.5 text-rf-green" />
            <span>AI-Driven Electronic Warfare Spectrum Intelligence & Adaptive Scan Strategy</span>
          </div>

          <h1 className="text-4xl sm:text-6xl font-display font-extrabold tracking-tight text-slate-100 leading-tight">
            SMART SCAN <br />
            <span className="bg-gradient-to-r from-rf-green-light via-rf-cyan-light to-slate-200 bg-clip-text text-transparent">
              COMMAND CENTER
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-400 font-sans max-w-2xl mx-auto leading-relaxed">
            Replace blind open-loop frequency sweeps with closed-loop reinforcement learning. 
            Dynamically allocate receiver dwell time to high-value hostile emitter bursts in dense electromagnetic environments.
          </p>

          {/* Action CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
            <button
              onClick={() => onGoToLogin()}
              className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-rf-green text-charcoal-950 font-mono font-bold text-sm shadow-glow-green hover:bg-rf-green-light transition-all transform hover:-translate-y-0.5"
            >
              <Lock className="w-4 h-4" />
              <span>AUTHENTICATE & ENTER COMMAND CENTER</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => onGoToLogin(SAMPLE_USERS[0])}
              className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-charcoal-900 hover:bg-charcoal-850 text-slate-200 border border-charcoal-700 hover:border-slate-500 font-mono text-sm font-semibold transition-all"
            >
              <Radio className="w-4 h-4 text-rf-cyan" />
              <span>SELECT DEMO OPERATOR ROLE</span>
            </button>
          </div>

          {/* Disclaimers & Dataset info */}
          <div className="pt-4 flex flex-wrap items-center justify-center gap-4 text-[11px] font-mono text-slate-500">
            <span className="flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-rf-cyan" />
              100,000 Observations Dataset
            </span>
            <span>•</span>
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-rf-green" />
              ε-Greedy Closed Loop Policy
            </span>
            <span>•</span>
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-rf-amber" />
              Simulated Decision-Support Prototype
            </span>
          </div>
        </div>
      </section>

      {/* 3. KEY METRICS & BENCHMARK HERO CARD */}
      <section className="py-12 px-6 max-w-7xl mx-auto">
        <div className="p-8 rounded-3xl bg-charcoal-900/90 border border-charcoal-750/90 shadow-2xl backdrop-blur-xl">
          <div className="text-center mb-8">
            <span className="text-[10px] font-mono uppercase font-bold text-rf-green tracking-wider">
              Empirical Performance Superiority
            </span>
            <h2 className="text-2xl font-display font-bold text-slate-100 mt-1">
              Why Cognitive Adaptive Scanning Outperforms Fixed Sweeping
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-2xl bg-charcoal-850/80 border border-rf-green-border shadow-glow-green text-center space-y-2">
              <div className="text-3xl sm:text-4xl font-mono font-black text-rf-green-light">
                +165%
              </div>
              <div className="text-xs font-mono uppercase text-slate-300 font-bold">
                Detection Rate Gain
              </div>
              <p className="text-xs text-slate-400 font-sans">
                Elevates target burst capture rate from 28.4% up to 74.5%+ across 20 frequency channels.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-charcoal-850/80 border border-rf-cyan-border shadow-glow-cyan text-center space-y-2">
              <div className="text-3xl sm:text-4xl font-mono font-black text-rf-cyan-light">
                -76%
              </div>
              <div className="text-xs font-mono uppercase text-slate-300 font-bold">
                Faster Intercept Time
              </div>
              <p className="text-xs text-slate-400 font-sans">
                Reduces time-to-first-intercept (TTFI) from 14.2 s down to 3.4 s, isolating threats immediately.
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-charcoal-850/80 border border-rf-amber-border shadow-glow-amber text-center space-y-2">
              <div className="text-3xl sm:text-4xl font-mono font-black text-rf-amber-light">
                3.8x
              </div>
              <div className="text-xs font-mono uppercase text-slate-300 font-bold">
                Energy & Dwell Efficiency
              </div>
              <p className="text-xs text-slate-400 font-sans">
                Requires only 1.8 dwells per positive detection instead of 7.0 blind dwells in open-loop sweeps.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. SAMPLE USER PROFILES SECTION */}
      <section className="py-12 px-6 max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-rf-cyan" />
              <h2 className="text-lg font-mono font-bold uppercase tracking-wider text-slate-200">
                OPERATIONAL DEMO ROLES & CLEARANCES
              </h2>
            </div>
            <p className="text-xs text-slate-400 font-sans mt-0.5">
              Select any pre-configured operational profile to log in with 1-click
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {SAMPLE_USERS.map((u) => (
            <div
              key={u.id}
              onClick={() => onGoToLogin(u)}
              className="p-5 rounded-2xl bg-charcoal-900 border border-charcoal-750 hover:border-rf-green-border hover:shadow-glow-green transition-all duration-200 cursor-pointer group flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="w-10 h-10 rounded-xl bg-charcoal-800 border border-charcoal-700 flex items-center justify-center font-mono font-bold text-sm text-rf-green-light group-hover:border-rf-green">
                    {u.avatarInitials}
                  </div>
                  <span className="text-[9px] font-mono px-2 py-0.5 rounded bg-charcoal-800 text-slate-300 border border-charcoal-700">
                    {u.clearance}
                  </span>
                </div>

                <div className="font-mono font-bold text-sm text-slate-100 group-hover:text-rf-green-light transition-colors">
                  {u.name}
                </div>
                <div className="text-xs text-slate-400 font-sans mt-0.5">{u.role}</div>
                <div className="text-[11px] text-rf-cyan font-mono mt-1">Callsign: {u.callsign}</div>
              </div>

              <div className="pt-4 mt-4 border-t border-charcoal-800 flex items-center justify-between text-xs font-mono text-slate-400">
                <span>{u.assignedStation}</span>
                <span className="text-rf-green flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                  <span>Sign In</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 5. SYSTEM ARCHITECTURE SUMMARY */}
      <section className="py-16 px-6 max-w-7xl mx-auto border-t border-charcoal-800/80">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
          <div className="space-y-4">
            <span className="text-[10px] font-mono uppercase font-bold text-rf-cyan tracking-wider">
              Mathematical Foundation
            </span>
            <h2 className="text-2xl sm:text-3xl font-display font-bold text-slate-100">
              The Cognitive Decision Loop
            </h2>
            <p className="text-sm text-slate-400 font-sans leading-relaxed">
              The system operates as a reinforcement feedback system. The ML Scheduler optimizes channel selection using empirical Bayes hit probability estimation combined with staleness bonuses to discover agile or migratory radar emitters.
            </p>

            <ul className="space-y-2.5 text-xs font-mono text-slate-300 pt-2">
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-rf-green shrink-0" />
                <span><strong>ML Scheduler</strong> decides: Which channel to dwell next.</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-rf-green shrink-0" />
                <span><strong>Detector</strong> decides: Does power exceed threshold (5.0 dB).</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-rf-green shrink-0" />
                <span><strong>Evaluator</strong> compares ground truth & issues reinforcement rewards.</span>
              </li>
            </ul>

            <div className="pt-4">
              <button
                onClick={() => onGoToLogin()}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rf-green/20 hover:bg-rf-green/30 text-rf-green border border-rf-green/40 font-mono text-xs font-bold transition-all"
              >
                <Lock className="w-3.5 h-3.5" />
                <span>Authenticate & Launch Command Center</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-charcoal-900 border border-charcoal-750 font-mono text-xs space-y-3 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-charcoal-750">
              <div className="flex items-center gap-2 text-slate-300 font-bold">
                <Terminal className="w-4 h-4 text-rf-green" />
                <span>PIPELINE TELEMETRY FLOW</span>
              </div>
              <span className="text-[10px] text-rf-green">v2.4 SIM</span>
            </div>

            <div className="space-y-2 text-[11px] text-slate-400">
              <div className="p-2.5 rounded-lg bg-charcoal-950 border border-charcoal-800">
                1. <span className="text-rf-cyan font-bold">RF Environment</span>: 20 Channels, pulsed emitters
              </div>
              <div className="p-2.5 rounded-lg bg-charcoal-950 border border-charcoal-800">
                2. <span className="text-slate-200 font-bold">Receiver</span>: Tunes to Band 7 (8.72 dB SNR)
              </div>
              <div className="p-2.5 rounded-lg bg-charcoal-950 border border-charcoal-800">
                3. <span className="text-rf-green-light font-bold">Detector</span>: ACTIVE (Threshold 5.0 dB exceeded)
              </div>
              <div className="p-2.5 rounded-lg bg-charcoal-950 border border-charcoal-800">
                4. <span className="text-rf-amber-light font-bold">Evaluator</span>: Verified HIT (+1.0 J Reward)
              </div>
              <div className="p-2.5 rounded-lg bg-charcoal-950 border border-charcoal-800">
                5. <span className="text-rf-green font-bold">ML Scheduler</span>: Updates Band 7 prior to 82%
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 6. FOOTER */}
      <footer className="py-8 px-6 border-t border-charcoal-800 text-center text-xs font-mono text-slate-500">
        <p>SMART SCAN COMMAND CENTER // Decision-Support Prototype using Simulated RF Environments.</p>
        <p className="mt-1 text-[11px] text-slate-600">Built for Electronic Warfare Spectrum Intelligence & Cognitive Strategy Research.</p>
      </footer>
    </div>
  );
};
