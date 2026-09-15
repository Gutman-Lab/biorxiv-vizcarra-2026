"""Configuration for the ICIAR 2018 BACH histology tile dataset."""

DATASET = "bach-dataset"

# Folder names under the BACH source root (mutually exclusive classes).
CLASSES = ["Benign", "InSitu", "Invasive", "Normal"]

EXCLUDE_IF_POSITIVE: tuple[str, ...] = ()

# Splits are assigned at load time and stored on each tile row.
SPLIT_FROM_COLUMN = True
DEFAULT_SPLIT_RATIOS = "80:20:20"
