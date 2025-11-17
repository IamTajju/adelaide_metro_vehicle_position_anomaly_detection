from enum import Enum


class AnomalyType(str, Enum):
    SPEED = "SPEED_ANOMALY"
    GEO_OUTLIER = "GEOGRAPHIC_OUTLIER"
    STATIONARY = "STATIONARY_TOO_LONG"


# Speed anomaly thresholds
MAX_SPEED_KMH = 100

# Geographic outlier thresholds
MIN_GLOBAL_HISTORY = 50
GEO_OUTLIER_THRESHOLD_KM = 50
GEO_NEIGHBOR_SAMPLE = 100

# Stationary detection thresholds
STATIONARY_WINDOW = 5
STATIONARY_DISTANCE_KM = 0.05

# Vehicle & global history settings
VEHICLE_HISTORY_WINDOW = 10
GLOBAL_POSITION_HISTORY = 1000

# Earth radius (Haversine)
DEFAULT_EARTH_RADIUS_KM = 6371

# ---- Half-Space / Isolation Forest Detector ---- #

# ---- Half-Space Trees (HST) ---- #
HST_NUM_TREES = 25            # ensemble size
HST_TREE_HEIGHT = 8           # depth of each tree (leaf count = 2**height)
HST_INIT_WINDOW = 500         # number of points to collect before initializing offsets
HST_FEATURE_DIM = 3           # lat, lon, speed
HST_DEFAULT_SCORE = 0.5       # neutral score before trees are ready
HST_RANDOM_SEED = 42          # reproducibility
