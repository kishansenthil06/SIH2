"""VENDORED, UNMODIFIED -- an epsilon-greedy multi-armed bandit scheduler.

Origin: `src/scheduler/ml_scheduler.py`, contributed separately from this
backend. Kept byte-for-byte apart from this header so it can be diffed against
its source; wrapping and adaptation live in `agent/policy_bandit.py`.

Why it is here: this is the *textbook* approach to "which band next", and the
architecture review this project is built on predicts it will underperform --

    "Bands turn on and off whether or not you are watching. That makes this a
     RESTLESS multi-armed bandit, not a standard one, and it is exactly why
     plain epsilon-greedy or UCB underperforms here -- those assume the arms
     wait for you."

`get_hit_rate` is a lifetime average over every scan of a band and never decays,
so a band that was busy early keeps its score forever. That is precisely the
stationarity assumption the quote names. Rather than assert the prediction, this
module is wired in as a **baseline** (`epsilon_greedy`) and measured head-to-head
on identical scenarios and seeds -- see the ablation table in README.md.

Note it chooses only *which band*; it has no notion of dwell, bandwidth, energy
or time, which is the other half of the action space this project cares about
(DESIGN.md section 3). The wrapper supplies fixed values for those and paces the
scanner so the comparison is fair on energy.
"""
import random


class MLScheduler:

    def __init__(self, frequency_bands, epsilon=0.2):

        self.frequency_bands = frequency_bands
        self.epsilon = epsilon

        # Number of times each band has been scanned
        self.scan_count = {
            band: 0
            for band in frequency_bands
        }

        # Number of successful detections
        self.hit_count = {
            band: 0
            for band in frequency_bands
        }

        # Total reward received
        self.total_reward = {
            band: 0.0
            for band in frequency_bands
        }

    # ----------------------------------------
    # Estimate success probability
    # ----------------------------------------

    def get_hit_rate(self, band):

        scans = self.scan_count[band]

        if scans == 0:
            return 0.0

        return (
            self.hit_count[band] / scans
        )

    # ----------------------------------------
    # Choose next frequency
    # ----------------------------------------

    def choose_band(self):

        # EXPLORATION
        if random.random() < self.epsilon:

            return random.choice(
                self.frequency_bands
            )

        # EXPLOITATION
        hit_rates = {
            band: self.get_hit_rate(band)
            for band in self.frequency_bands
        }

        best_band = max(
            hit_rates,
            key=hit_rates.get
        )

        return best_band

    # ----------------------------------------
    # Learn from result
    # ----------------------------------------

    def update(self, band, reward, hit):

        self.scan_count[band] += 1

        self.total_reward[band] += reward

        if hit:
            self.hit_count[band] += 1

    # ----------------------------------------
    # Display learned knowledge
    # ----------------------------------------

    def get_statistics(self):

        statistics = {}

        for band in self.frequency_bands:

            statistics[band] = {
                "scans": self.scan_count[band],
                "hits": self.hit_count[band],
                "hit_rate": self.get_hit_rate(band),
                "reward": self.total_reward[band]
            }

        return statistics