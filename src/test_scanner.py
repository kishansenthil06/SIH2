from environment import RFEnvironment
from receiver import Receiver
from scanner import BaselineScanner


DATASET_PATH = "data/prototype/temporary_rf_dataset.csv"


environment = RFEnvironment(DATASET_PATH)
receiver = Receiver(environment)
scanner = BaselineScanner(
    receiver=receiver,
    num_bands=20,
)


print("\n--- Baseline Scanner ---")

for scan_number in range(10):
    observation = scanner.scan(scan_number)

    print(
        f"Scan {scan_number + 1}: "
        f"time={observation['time_slot']}, "
        f"band={observation['frequency_band']}, "
        f"power={observation['signal_power']:.3f}"
    )