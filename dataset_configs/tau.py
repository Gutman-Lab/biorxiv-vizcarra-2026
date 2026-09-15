"""Configuration for the Vizcarra-2023 tau tile dataset."""

DATASET = "tau-dataset"

CLASSES = ["pre_nft", "inft"]

# No Abeta-style notsure/flag columns in vizcarra metadata.csv.
EXCLUDE_IF_POSITIVE: tuple[str, ...] = ()

# Splits are written into the tiles table from metadata.csv (train/validation/hold-out).
SPLIT_FROM_COLUMN = True
DEFAULT_SPLIT_RATIOS = None
