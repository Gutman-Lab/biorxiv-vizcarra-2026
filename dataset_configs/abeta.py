"""Configuration for the ABeta tile dataset."""

DATASET = "abeta-dataset"

CLASSES = ["cored", "diffuse", "CAA"]

# CSV columns required beyond CLASSES (rows with these > 0 are excluded).
EXCLUDE_IF_POSITIVE = ("notsure", "flag")

# ABeta leaves split null; linear probe assigns WSI-level splits at eval time.
SPLIT_FROM_COLUMN = False
DEFAULT_SPLIT_RATIOS = (0.6, 0.2, 0.2)
