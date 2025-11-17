# main.py (Updated for All Detectors and Compute Times)

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
from lof_detector import LOFAnomalyDetector  # Now with compute time tracking
from decode_gtfs_feed import decode_gtfs_feed
from rule_based_anomaly_detector import RuleBasedAnomalyDetector
from csv_anomaly_formatter import CSVAnomalyFormatter
from stream_logger import StreamDataLogger

# ==================== Main Pipeline ====================


def main():
    print("=" * 80)
    print("Adelaide Metro Real-time Anomaly Detection System (Multi-Algorithm)")
    print("=" * 80)

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

    # Build the pipeline
    decoded_stream = stream.flat_map(
        decode_gtfs_feed,
        output_type=Types.STRING()
    )

    # Print data logs and summary statistics
    decoded_stream.map(
        StreamDataLogger()
    )

    enriched_stream = decoded_stream

    # Append Vehicle speed to Stream
    enriched_stream = decoded_stream.map(
        AppendSpeedToStream(),
        output_type=Types.STRING()
    )

    # A. Apply Rule-Based Detector (adds 'anomalies' list and 'is_anomaly' flag)
    enriched_stream = enriched_stream.map(
        RuleBasedAnomalyDetector(),
        output_type=Types.STRING()
    )

    # B. Apply Half-Space Detector (adds 'half_space_score' and 'half_space_compute_ms')
    enriched_stream = enriched_stream.map(
        HalfSpaceAnomalyDetector(),
        output_type=Types.STRING()
    )

    # C. Apply Isolation Forest Detector (adds 'iforest_score' and 'iforest_compute_ms')
    enriched_stream = enriched_stream.map(
        IsolationForestAnomalyDetector(),
        output_type=Types.STRING()
    )

    # D. Apply LOF Detector (adds 'lof_score' and 'lof_compute_ms')
    enriched_stream = enriched_stream.map(
        LOFAnomalyDetector(),
        output_type=Types.STRING()
    )

    # E. Apply DBSCAN Detector (adds 'dbscan_is_anomaly' flag and 'dbscan_compute_ms')
    enriched_stream = enriched_stream.map(
        DBSCANAnomalyDetector(),
        output_type=Types.STRING()
    )

    # Filter and format output to CSV (only anomalies included)
    csv_output = enriched_stream.map(
        CSVAnomalyFormatter(),
        output_type=Types.STRING()
    )

    # --- CSV File Sink Setup ---
    # Define the comprehensive CSV Header
    csv_header = (
        "timestamp,vehicle_id,latitude,longitude,rule_anomaly,rule_anomaly_types,"
        "half_space_score,half_space_compute_ms,"
        "iforest_score,iforest_compute_ms,"
        "lof_score,lof_compute_ms,"
        "dbscan_label,dbscan_compute_ms"
    )

    sink = (FileSink
            .for_row_format('/Users/tajju/Desktop/Assignments/COMP7707/ahme0423_A3/a3_prototype/output', Encoder.simple_string_encoder("UTF-8"))
            .with_rolling_policy(RollingPolicy.default_rolling_policy(
                part_size=1024 ** 3,
                rollover_interval=15 * 60 * 1000,
                inactivity_interval=5 * 60 * 1000))
            .build())

    # Filter out empty strings (non-anomalous points) before sinking
    csv_output.filter(lambda x: len(x) > 0).sink_to(sink)

    print("\n🚀 Starting anomaly detection pipeline...")
    print(f"✅ CSV Header (Must be manually added to file): {csv_header}")
    print("👀 Monitoring for anomalies and writing results to CSV...\n")

    # Execute
    env.execute("Adelaide Metro Anomaly Detection (Multi-Algorithm)")


if __name__ == "__main__":
    main()
