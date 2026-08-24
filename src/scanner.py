import random


class BaselineScanner:
    """
    Traditional sequential scanner.

    The scanner visits frequency bands in order:
    1 -> 2 -> 3 -> ... -> N -> 1 -> ...
    """

    def __init__(self, receiver, num_bands=20):
        self.receiver = receiver
        self.num_bands = num_bands

    def scan(self, scan_number):
        # For the prototype:
        # one scan = one simulation time step.
        time_slot = scan_number

        # Cycle through bands 1 -> 20 -> 1 -> ...
        frequency_band = (scan_number % self.num_bands) + 1

        observation = self.receiver.scan(
            time_slot=time_slot,
            frequency_band=frequency_band,
        )

        return observation


class RandomScanner:
    """
    Randomized scanner exploring uniform frequency bands.
    """

    def __init__(self, receiver, num_bands=20, seed=42):
        self.receiver = receiver
        self.num_bands = num_bands
        self._rng = random.Random(seed)

    def scan(self, scan_number):
        time_slot = scan_number
        frequency_band = self._rng.randint(1, self.num_bands)

        return self.receiver.scan(
            time_slot=time_slot,
            frequency_band=frequency_band,
        )


class AdaptiveScanner:
    """
    Heuristic adaptive scanner that tracks recent activity and stays on or
    re-checks active bands with high priority while periodically exploring.
    """

    def __init__(self, receiver, num_bands=20, explore_rate=0.2, seed=42):
        self.receiver = receiver
        self.num_bands = num_bands
        self.explore_rate = explore_rate
        self._rng = random.Random(seed)
        self.band_activity = {b: 0.0 for b in range(1, num_bands + 1)}
        self.last_observation = None

    def scan(self, scan_number):
        time_slot = scan_number

        # Update activity score from last observation
        if self.last_observation:
            band = self.last_observation["frequency_band"]
            power = self.last_observation.get("signal_power", 0.0)
            if power > 5.0:
                self.band_activity[band] = self.band_activity.get(band, 0.0) * 0.8 + 2.0
            else:
                self.band_activity[band] = self.band_activity.get(band, 0.0) * 0.7

        # Choose next band: explore or exploit active bands
        if self._rng.random() < self.explore_rate or not any(v > 0.5 for v in self.band_activity.values()):
            frequency_band = (scan_number % self.num_bands) + 1
        else:
            frequency_band = max(self.band_activity.items(), key=lambda x: x[1])[0]

        observation = self.receiver.scan(
            time_slot=time_slot,
            frequency_band=frequency_band,
        )
        self.last_observation = observation
        return observation


def make_scanner(strategy: str, receiver, num_bands: int = 20, **kwargs):
    """Factory to create a scanner instance based on strategy name."""
    strat = str(strategy).lower().strip()
    if strat in ("random", "rand"):
        return RandomScanner(receiver, num_bands=num_bands, **kwargs)
    elif strat in ("adaptive", "smart", "heuristic"):
        return AdaptiveScanner(receiver, num_bands=num_bands, **kwargs)
    else:
        return BaselineScanner(receiver, num_bands=num_bands)