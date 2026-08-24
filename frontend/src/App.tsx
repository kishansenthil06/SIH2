import { useState } from 'react';
import { FloatingStaircaseNav, type TabId } from './components/FloatingStaircaseNav';
import { BandDetailModal } from './components/BandDetailModal';
import { useSimulation } from './hooks/useSimulation';
import type { UserProfile } from './types/auth';


import { LandingPage } from './pages/LandingPage';
import { LoginPage } from './pages/LoginPage';
import { CommandCenter } from './pages/CommandCenter';
import { SpectrumIntelligence } from './pages/SpectrumIntelligence';
import { MLScheduler } from './pages/MLScheduler';
import { Performance } from './pages/Performance';
import { Simulation } from './pages/Simulation';

type AppView = 'landing' | 'login' | 'app';

export function App() {
  const [currentView, setCurrentView] = useState<AppView>('landing');
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('command-center');
  const [loginSelectedUser, setLoginSelectedUser] = useState<UserProfile | undefined>(undefined);

  const {
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
  } = useSimulation();

  const handleGoToLogin = (user?: UserProfile) => {
    if (user) setLoginSelectedUser(user);
    setCurrentView('login');
  };

  const handleLoginSuccess = (user: UserProfile) => {
    setCurrentUser(user);
    setCurrentView('app');
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setCurrentView('login');
  };

  const handleGoToLanding = () => {
    setCurrentView('landing');
  };

  // 1. Landing Page View
  if (currentView === 'landing') {
    return (
      <LandingPage
        onGoToLogin={handleGoToLogin}
      />
    );
  }

  // 2. Login Page View or Auth Guard
  if (currentView === 'login' || !currentUser) {
    return (
      <LoginPage
        initialUser={loginSelectedUser}
        onLoginSuccess={handleLoginSuccess}
        onBackToLanding={handleGoToLanding}
      />
    );
  }


  // 3. Main Operational Command Center Application View
  return (
    <div className="min-h-screen bg-charcoal-950 text-slate-100 font-sans flex flex-col relative">
      {/* Floating SMART SCAN Logo (Top-Left) with Staircase Cascade & Floating Controls (Top-Right) */}
      <FloatingStaircaseNav
        activeTab={activeTab}
        onTabChange={setActiveTab}
        timeSlot={currentScan.time_slot}
        isRunning={isRunning}
        demoMode={config.demo_mode}
        onToggleDemoMode={toggleDemoMode}
        onStart={startSimulation}
        onPause={pauseSimulation}
        onReset={resetSimulation}
        onStep={stepSimulation}
        currentUser={currentUser}
        onLogout={handleLogout}
        onGoToLanding={handleGoToLanding}
      />

      {/* Main Full-Width Content Container */}
      <main className="flex-1 px-6 pt-20 pb-8 max-w-7xl w-full mx-auto">
        {activeTab === 'command-center' && (
          <CommandCenter
            isRunning={isRunning}
            demoMode={config.demo_mode}
            demoNarrative={demoNarrative}
            currentScan={currentScan}
            schedulerDecision={schedulerDecision}
            bands={bands}
            history={history}
            onSelectBand={setSelectedBandDetail}
          />
        )}

        {activeTab === 'spectrum-intel' && (
          <SpectrumIntelligence
            bands={bands}
            timeSlot={currentScan.time_slot}
            onSelectBand={setSelectedBandDetail}
          />
        )}

        {activeTab === 'ml-scheduler' && (
          <MLScheduler
            decision={schedulerDecision}
            timeline={timeline}
            rewardTrajectory={rewardTrajectory}
          />
        )}

        {activeTab === 'performance' && (
          <Performance performance={performance} />
        )}

        {activeTab === 'simulation' && (
          <Simulation
            isRunning={isRunning}
            config={config}
            currentScan={currentScan}
            eventLogs={eventLogs}
            onStart={startSimulation}
            onPause={pauseSimulation}
            onReset={resetSimulation}
            onStep={stepSimulation}
            onUpdateStrategy={updateStrategy}
            onUpdateSpeed={updateSpeed}
            onUpdateEpsilon={updateEpsilon}
          />
        )}
      </main>

      {/* Inspect Band Detail Modal */}
      <BandDetailModal
        band={selectedBandDetail}
        onClose={() => setSelectedBandDetail(null)}
      />
    </div>
  );
}

export default App;
