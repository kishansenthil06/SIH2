class PowerThresholdDetector:
    """
    Simple rule-based detector.

    If received signal power is above the threshold,
    classify the observation as an active signal.
    """

    def __init__(self, threshold=5.0):
        self.threshold = threshold

    def predict(self, observation):
        signal_power = observation["signal_power"]

        prediction = signal_power > self.threshold

        return prediction