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