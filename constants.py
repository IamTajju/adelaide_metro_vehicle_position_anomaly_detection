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

RANDOM_STATE = 42
UNIFIED_WARMUP = 500

# ---- Half-Space Trees (HST) ---- #
HST_NUM_TREES = 25
HST_TREE_HEIGHT = 15
HST_INIT_WINDOW = UNIFIED_WARMUP

# ----  Isolation Forest  ---- #
IFOREST_N_ESTIMATORS = 100
IFOREST_MAX_SAMPLES = 256
IFOREST_WINDOW_SIZE = UNIFIED_WARMUP
IFOREST_RETRAIN_INTERVAL = 100


# ----  DBSCAN  ---- #
DBSCAN_MIN_SAMPLES = 4              # 3 dimensions + 1
DBSCAN_EPS = 0.5                    # Epsilon radius (may need tuning)
DBSCAN_WINDOW_SIZE = UNIFIED_WARMUP
DBSCAN_UPDATE_INTERVAL = 100        # Recluster frequency

# ----  LOF  ---- #
LOF_N_NEIGHBORS = 20                # Standard recommendation from sklearn
LOF_WINDOW_SIZE = UNIFIED_WARMUP               # Match other algorithms
LOF_UPDATE_INTERVAL = 100           # Retrain frequency
LOF_MIN_SAMPLES = UNIFIED_WARMUP
