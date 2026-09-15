"""Configuration for the TCGA TIL tile dataset (Abousamra et al. / Zenodo)."""

from config import TEST_SPLIT, TRAIN_SPLIT, TUNE_SPLIT

DATASET = "till-dataset"

# Positive class only. All-zero label vectors are the implicit ``negative`` class
# (see ``utils.linear_probe.classes.training_class_names`` / ``labels_to_class_index``).
CLASSES = ["til-positive"]

# On-disk / metadata label for negatives (not part of CLASSES).
NEGATIVE_LABEL = "til-negative"

EXCLUDE_IF_POSITIVE: tuple[str, ...] = ()

# Splits come from metadata ``partition`` (train/val/test → train/validation/hold-out).
# No load-time or default ratio assignment — partitions are fixed in the CSV.
SPLIT_FROM_COLUMN = True
DEFAULT_SPLIT_RATIOS = None

# Metadata CSV must live under the dataset root (no override path).
METADATA_FILENAME = "images-tcga-tils-metadata.csv"
IMAGES_DIRNAME = "images-tcga-tils"

# Map dataset partition names onto Pixeltable split values.
PARTITION_TO_SPLIT = {
    "train": TRAIN_SPLIT,
    "val": TUNE_SPLIT,
    "test": TEST_SPLIT,
}
