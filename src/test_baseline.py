from environment import RFEnvironment
from receiver import Receiver
from scanner import BaselineScanner
from detector import PowerThresholdDetector
from evaluator import Evaluator


DATASET_PATH = "data/prototype/temporary_rf_dataset.csv"


environment = RFEnvironment(DATASET_PATH)

receiver = Receiver(environment)

scanner = BaselineScanner(
    receiver=receiver,
    num_bands=20,
)

detector = PowerThresholdDetector(
    threshold=5.0,
)

evaluator = Evaluator(environment)


print("\n--- Baseline System ---")

NUM_SCANS = 100


for scan_number in range(NUM_SCANS):

    # 1. Scanner chooses the next band.
    observation = scanner.scan(scan_number)

    # 2. Detector predicts whether a signal exists.
    prediction = detector.predict(observation)

    # 3. Evaluator compares prediction with hidden truth.
    result = evaluator.evaluate(
        time_slot=observation["time_slot"],
        frequency_band=observation["frequency_band"],
        prediction=prediction,
    )

    print(
        f"Scan {scan_number + 1:3d} | "
        f"Time {observation['time_slot']:3d} | "
        f"Band {observation['frequency_band']:2d} | "
        f"Power {observation['signal_power']:7.3f} | "
        f"Prediction {prediction!s:5s} | "
        f"{result['result']}"
    )


print("\n--- Baseline Metrics ---")

metrics = evaluator.metrics()

for key, value in metrics.items():
    print(f"{key}: {value}")