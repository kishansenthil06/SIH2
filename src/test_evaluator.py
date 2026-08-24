from environment import RFEnvironment
from evaluator import Evaluator


DATASET_PATH = "data/prototype/temporary_rf_dataset.csv"


environment = RFEnvironment(DATASET_PATH)
evaluator = Evaluator(environment)


print("\n--- Test 1: Correct ACTIVE prediction ---")

result = evaluator.evaluate(
    time_slot=0,
    frequency_band=3,
    prediction=True,
)

print(result)


print("\n--- Test 2: Correct INACTIVE prediction ---")

result = evaluator.evaluate(
    time_slot=0,
    frequency_band=1,
    prediction=False,
)

print(result)


print("\n--- Test 3: Wrong prediction ---")

result = evaluator.evaluate(
    time_slot=0,
    frequency_band=1,
    prediction=True,
)

print(result)


print("\n--- Metrics ---")

print(evaluator.metrics())