# main.py
from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.common.typeinfo import Types
from pyflink.java_gateway import get_gateway
from pyflink.datastream import DataStream
from pyflink.datastream.connectors.file_system import FileSink, RollingPolicy
from pyflink.common import Encoder
import os

# ----------------- Import Custom Detectors -----------------
from append_speed_to_stream import AppendSpeedToStream
from dbscan_detector import DBSCANAnomalyDetector
from half_space_detector import HalfSpaceAnomalyDetector
from isolation_forest_detector import IsolationForestAnomalyDetector
from lof_detector import LOFAnomalyDetector
from decode_gtfs_feed import decode_gtfs_feed
from rule_based_anomaly_detector import RuleBasedAnomalyDetector
from csv_anomaly_formatter import CSVAnomalyFormatter
from stream_logger import StreamDataLogger

# Import constants for warm-up info
from constants import (
    HST_INIT_WINDOW,
    IFOREST_WINDOW_SIZE,
    DBSCAN_WINDOW_SIZE,
    LOF_MIN_SAMPLES,
)

# ==================== Main Pipeline ====================


def main():
    print("=" * 80)
    print("Adelaide Metro Real-time Anomaly Detection System (Multi-Algorithm)")
    print("=" * 80)

    # Display warm-up periods for all algorithms
    print("\n📊 Algorithm Warm-up Periods:")
    print(f"   - Half-Space Trees (HST):  {HST_INIT_WINDOW} samples")
    print(f"   - Isolation Forest:        {IFOREST_WINDOW_SIZE} samples")
    print(f"   - DBSCAN:                  {DBSCAN_WINDOW_SIZE} samples")
    print(f"   - LOF:                     {LOF_MIN_SAMPLES} samples")
    max_warmup = max(HST_INIT_WINDOW, IFOREST_WINDOW_SIZE,
                     DBSCAN_WINDOW_SIZE, LOF_MIN_SAMPLES)
    print(f"   ⚠️  Valid comparisons start after sample {max_warmup}\n")

    # Setup Flink environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    env.set_parallelism(1)

    # Add your compiled JAR
    jar_path = os.path.abspath("PollingVehicleSource.jar")
    env.add_jars(f"file://{jar_path}")

    # Create Java source and wrap to Python DataStream
    gateway = get_gateway()
    j_env = env._j_stream_execution_environment
    j_source = gateway.jvm.PollingVehicleSource()
    j_stream = j_env.addSource(j_source)
    stream = DataStream(j_stream)

    # ==================== Pipeline Stages ====================

    # Stage 1: Decode GTFS feed
    decoded_stream = stream.flat_map(
        decode_gtfs_feed,
        output_type=Types.STRING()
    )

    # Stage 2: Log data (optional monitoring)
    decoded_stream.map(StreamDataLogger())

    # Stage 3: Enrich with vehicle speed
    enriched_stream = decoded_stream.map(
        AppendSpeedToStream(),
        output_type=Types.STRING()
    )

    # Stage 4: Apply Rule-Based Detector
    # (adds 'anomalies' list and 'is_anomaly' flag)
    enriched_stream = enriched_stream.map(
        RuleBasedAnomalyDetector(),
        output_type=Types.STRING()
    )

    # ==================== ML Anomaly Detectors ====================
    # All detectors process sequentially, each adding their scores
    # Note: Order doesn't affect results since each detector is independent

    # Stage 5A: Half-Space Trees (streaming algorithm)
    enriched_stream = enriched_stream.map(
        HalfSpaceAnomalyDetector(),
        output_type=Types.STRING()
    )

    # Stage 5B: Isolation Forest (sliding window)
    enriched_stream = enriched_stream.map(
        IsolationForestAnomalyDetector(),
        output_type=Types.STRING()
    )

    # Stage 5C: Local Outlier Factor (sliding window)
    enriched_stream = enriched_stream.map(
        LOFAnomalyDetector(),
        output_type=Types.STRING()
    )

    # Stage 5D: DBSCAN (sliding window clustering)
    enriched_stream = enriched_stream.map(
        DBSCANAnomalyDetector(),
        output_type=Types.STRING()
    )

    # ==================== Output Formatting ====================

    # Stage 6: Format to CSV (filters for valid scores)
    csv_output = enriched_stream.map(
        CSVAnomalyFormatter(),
        output_type=Types.STRING()
    )

    # --- CSV File Sink Setup ---
    csv_header = (
        "timestamp,vehicle_id,route_id,latitude,longitude,speed_kmh,"
        "rule_anomaly,rule_anomaly_types,"
        "half_space_score,half_space_compute_ms,"
        "iforest_score,iforest_compute_ms,"
        "lof_score,lof_compute_ms,"
        "dbscan_score,dbscan_compute_ms"
    )

    output_path = '/Users/tajju/Desktop/Assignments/COMP7707/ahme0423_A3/a3_prototype/output'

    sink = (FileSink
            .for_row_format(output_path, Encoder.simple_string_encoder("UTF-8"))
            .with_rolling_policy(RollingPolicy.default_rolling_policy(
                part_size=1024 ** 3,              # 1GB file size
                rollover_interval=15 * 60 * 1000,  # 15 minutes
                inactivity_interval=5 * 60 * 1000))  # 5 minutes
            .build())

    # Filter out empty strings (warmup samples or invalid data)
    csv_output.filter(lambda x: len(x) > 0).sink_to(sink)

    # ==================== Execution ====================

    print("\n🚀 Starting anomaly detection pipeline...")
    print(f"📁 Output directory: {output_path}")
    print(f"✅ CSV Header (add manually to output file):")
    print(f"   {csv_header}")
    print("\n⏳ Pipeline stages:")
    print("   1. Decode GTFS feed")
    print("   2. Enrich with speed calculations")
    print("   3. Rule-based detection")
    print("   4. ML detectors (HST → IForest → LOF → DBSCAN)")
    print("   5. Format and write to CSV")
    print("\n👀 Monitoring for anomalies...\n")

    # Execute pipeline
    env.execute("Adelaide Metro Anomaly Detection (Multi-Algorithm)")


if __name__ == "__main__":
    main()
