from environment import RFEnvironment
from receiver import Receiver
from detector import PowerThresholdDetector


DATASET_PATH = "data/prototype/temporary_rf_dataset.csv"


environment = RFEnvironment(DATASET_PATH)
receiver = Receiver(environment)
detector = PowerThresholdDetector(threshold=5.0)


print("\n--- Detector Test ---")

for band in [1, 3, 5, 9]:
    observation = receiver.scan(
        time_slot=0,
        frequency_band=band,
    )

    prediction = detector.predict(observation)

    print(
        f"Band {band}: "
        f"power={observation['signal_power']:.3f}, "
        f"prediction={prediction}"
    )