class Evaluator:
    def __init__(self, environment):
        self.environment = environment

        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

    def evaluate(self, time_slot, frequency_band, prediction):
        """
        Compare the model's prediction against the hidden
        ground truth from the RF environment.
        """

        # Get ground truth from the environment.
        observation = self.environment.scan(
            time_slot=time_slot,
            frequency_band=frequency_band,
        )

        ground_truth = observation["hit"]

        # Convert prediction to boolean.
        prediction = bool(prediction)

        # Determine result.
        if prediction and ground_truth:
            result = "true_positive"
            self.true_positive += 1

        elif prediction and not ground_truth:
            result = "false_positive"
            self.false_positive += 1

        elif not prediction and ground_truth:
            result = "false_negative"
            self.false_negative += 1

        else:
            result = "true_negative"
            self.true_negative += 1

        return {
            "time_slot": time_slot,
            "frequency_band": frequency_band,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "result": result,
        }

    def metrics(self):
        """
        Return evaluation metrics accumulated so far.
        """

        total = (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

        if total == 0:
            return {
                "total_scans": 0,
                "accuracy": 0.0,
                "detection_rate": 0.0,
                "false_alarm_rate": 0.0,
            }

        accuracy = (
            self.true_positive + self.true_negative
        ) / total

        actual_active = self.true_positive + self.false_negative

        actual_inactive = self.true_negative + self.false_positive

        detection_rate = (
            self.true_positive / actual_active
            if actual_active > 0
            else 0.0
        )

        false_alarm_rate = (
            self.false_positive / actual_inactive
            if actual_inactive > 0
            else 0.0
        )

        precision = (
            self.true_positive / (self.true_positive + self.false_positive)
            if (self.true_positive + self.false_positive) > 0
            else 0.0
        )

        f1_score = (
            2 * (precision * detection_rate) / (precision + detection_rate)
            if (precision + detection_rate) > 0
            else 0.0
        )

        return {
            "total_scans": total,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "accuracy": round(accuracy, 4),
            "detection_rate": round(detection_rate, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1_score, 4),
        }

    def reset(self):
        """Reset accumulated confusion matrix counts."""
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0