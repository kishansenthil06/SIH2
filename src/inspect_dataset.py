from pathlib import Path

from turing_deinterleaving_challenge import PulseTrain


# --------------------------------------------------
# 1. Locate our dataset file
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "sample" / "config_0.h5"

print("=" * 60)
print("TSRD DATASET INSPECTION")
print("=" * 60)

print(f"\nDataset file:")
print(DATA_FILE)

print(f"File exists: {DATA_FILE.exists()}")


# --------------------------------------------------
# 2. Load the pulse train
# --------------------------------------------------

print("\nLoading pulse train...")

pulse_train = PulseTrain.load(DATA_FILE)

print("Pulse train loaded successfully!")


# --------------------------------------------------
# 3. Inspect the main components
# --------------------------------------------------

print("\n" + "=" * 60)
print("MAIN COMPONENTS")
print("=" * 60)

print("\nData:")
print(type(pulse_train.data))

print("\nLabels:")
print(type(pulse_train.labels))

print("\nMetadata:")
print(type(pulse_train.metadata))


# --------------------------------------------------
# 4. Inspect shapes
# --------------------------------------------------

print("\n" + "=" * 60)
print("SHAPES")
print("=" * 60)

try:
    print(f"\nData shape: {pulse_train.data.shape}")
except AttributeError:
    print("\nData does not have a standard .shape attribute.")

try:
    print(f"Labels shape: {pulse_train.labels.shape}")
except AttributeError:
    print("\nLabels do not have a standard .shape attribute.")


# --------------------------------------------------
# 5. Show a few PDWs
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIRST FEW DATA ENTRIES")
print("=" * 60)

print(pulse_train.data[:5])


# --------------------------------------------------
# 6. Show a few labels
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIRST FEW LABELS")
print("=" * 60)

print(pulse_train.labels[:20])


# --------------------------------------------------
# 7. Metadata
# --------------------------------------------------

print("\n" + "=" * 60)
print("METADATA")
print("=" * 60)

print(pulse_train.metadata)