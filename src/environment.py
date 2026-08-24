import csv
from pathlib import Path


class RFEnvironment:
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = "data/prototype/temporary_rf_dataset.csv"
        
        path = Path(csv_path)
        if not path.is_absolute() and not path.exists():
            # Try resolving relative to repository root
            repo_root = Path(__file__).resolve().parent.parent
            candidate = repo_root / path
            if candidate.exists():
                path = candidate

        self.csv_path = Path(path)
        self.data = {}
        self.time_slots = set()
        self.frequency_bands = set()
        self._stats = None

        self._load_dataset()

    def _load_dataset(self):
        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                time_slot = int(row["time_slot"])
                frequency_band = int(row["frequency_band"])

                self.time_slots.add(time_slot)
                self.frequency_bands.add(frequency_band)

                self.data[(time_slot, frequency_band)] = {
                    "emitter_active_truth": int(row["emitter_active_truth"]),
                    "signal_power": float(row["signal_power"]),
                    "pulse_width": float(row["pulse_width"]),
                    "angle_of_arrival": (
                        float(row["angle_of_arrival"])
                        if row["angle_of_arrival"]
                        else None
                    ),
                }

        print(f"Loaded {len(self.data)} RF observations across {len(self.time_slots)} time slots.")

    def scan(self, time_slot, frequency_band):
        key = (time_slot, frequency_band)

        if key not in self.data:
            raise ValueError(
                f"No data found for time_slot={time_slot}, "
                f"frequency_band={frequency_band}"
            )

        observation = self.data[key]

        return {
            "time_slot": time_slot,
            "frequency_band": frequency_band,
            "hit": observation["emitter_active_truth"] == 1,
            "signal_power": observation["signal_power"],
            "pulse_width": observation["pulse_width"],
            "angle_of_arrival": observation["angle_of_arrival"],
        }

    def get_time_slot_observations(self, time_slot: int) -> list[dict]:
        """Returns all frequency band observations for a single time slot."""
        obs = []
        for band in sorted(self.frequency_bands):
            key = (time_slot, band)
            if key in self.data:
                d = self.data[key]
                obs.append({
                    "time_slot": time_slot,
                    "frequency_band": band,
                    "hit": d["emitter_active_truth"] == 1,
                    "signal_power": d["signal_power"],
                    "pulse_width": d["pulse_width"],
                    "angle_of_arrival": d["angle_of_arrival"],
                })
        return obs

    def get_stats(self) -> dict:
        """Returns statistical summary of the loaded dataset."""
        if self._stats is not None:
            return self._stats

        n_obs = len(self.data)
        if n_obs == 0:
            return {}

        active_count = sum(1 for v in self.data.values() if v["emitter_active_truth"] == 1)
        powers = [v["signal_power"] for v in self.data.values()]
        pws = [v["pulse_width"] for v in self.data.values()]

        self._stats = {
            "dataset_file": str(self.csv_path),
            "total_observations": n_obs,
            "num_time_slots": len(self.time_slots),
            "num_frequency_bands": len(self.frequency_bands),
            "frequency_bands": sorted(self.frequency_bands),
            "active_signals_count": active_count,
            "inactive_signals_count": n_obs - active_count,
            "active_ratio": round(active_count / n_obs, 4),
            "signal_power_mean": round(sum(powers) / n_obs, 4),
            "signal_power_min": round(min(powers), 4),
            "signal_power_max": round(max(powers), 4),
            "pulse_width_mean": round(sum(pws) / n_obs, 4),
            "pulse_width_max": round(max(pws), 4),
        }
        return self._stats