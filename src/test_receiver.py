from environment import RFEnvironment
from receiver import Receiver


DATASET_PATH = "data/prototype/temporary_rf_dataset.csv"


environment = RFEnvironment(DATASET_PATH)
receiver = Receiver(environment)


result = receiver.scan(
    time_slot=0,
    frequency_band=3,
)

print("Receiver observation:")
print(result)