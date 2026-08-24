class Receiver:
    def __init__(self, environment):
        self.environment = environment

    def scan(self, time_slot, frequency_band):
        observation = self.environment.scan(
            time_slot=time_slot,
            frequency_band=frequency_band,
        )

        return {
            "time_slot": observation["time_slot"],
            "frequency_band": observation["frequency_band"],
            "signal_power": observation["signal_power"],
            "pulse_width": observation["pulse_width"],
            "angle_of_arrival": observation["angle_of_arrival"],
        }


if __name__ == "__main__":
    try:
        from src.environment import RFEnvironment
    except ImportError:
        from environment import RFEnvironment

    env = RFEnvironment()
    rx = Receiver(env)
    obs = rx.scan(time_slot=0, frequency_band=3)
    print("\n--- Receiver Demonstration ---")
    print(f"Time Slot       : {obs['time_slot']}")
    print(f"Frequency Band  : {obs['frequency_band']}")
    print(f"Signal Power    : {obs['signal_power']:.3f} dB")
    print(f"Pulse Width     : {obs['pulse_width']:.3f} µs")
    print(f"Angle of Arrival: {obs['angle_of_arrival']}")
    print(f"Full Observation: {obs}\n")