from environment import RFEnvironment


DATASET_PATH = "data/prototype/temporary_rf_dataset.csv"


environment = RFEnvironment(DATASET_PATH)

observation = environment.scan(
    time_slot=0,
    frequency_band=3,
)

print("Observation:")
print(observation)