from pyflink.datastream.functions import MapFunction
from datetime import datetime
import json


class StreamDataLogger(MapFunction):

    def __init__(self):
        self.call_count = 0
        self.seen_vehicle_ids = set()
        self.vehicle_message_counts = {}

        # batch tracking
        self.current_batch_timestamp = None
        self.records_in_batch = 0

    def _print_batch_stats(self, timestamp):
        total_vehicles = len(self.seen_vehicle_ids)
        total_messages = sum(self.vehicle_message_counts.values())
        avg = total_messages / total_vehicles if total_vehicles > 0 else 0

        print("\n" + "="*70)
        print(f"STREAM DATA LOGGER — BATCH @ {timestamp}")
        print("-"*70)
        print(
            f"Records in batch                     : {self.records_in_batch}")
        print(f"Total lifetime unique vehicles       : {total_vehicles}")
        print(f"Average data points per vehicle      : {avg:.2f}")
        print("="*70 + "\n")

    def map(self, json_str: str) -> str:
        self.call_count += 1

        # detect batch timestamp (from system time)
        now_obj = datetime.now()
        now_str = now_obj.strftime("%Y-%m-%d %H:%M:%S")

        # first record ever → initialize batch
        if self.current_batch_timestamp is None:
            self.current_batch_timestamp = now_str

        # new batch detected when timestamp changes
        if now_str != self.current_batch_timestamp:
            self._print_batch_stats(self.current_batch_timestamp)
            # reset batch
            self.current_batch_timestamp = now_str
            self.records_in_batch = 0

        self.records_in_batch += 1

        # normal per-vehicle processing
        try:
            vehicle = json.loads(json_str)

            vehicle_id = vehicle.get("vehicle_id")

            if vehicle_id:
                if vehicle_id not in self.seen_vehicle_ids:
                    self.seen_vehicle_ids.add(vehicle_id)
                self.vehicle_message_counts[vehicle_id] = \
                    self.vehicle_message_counts.get(vehicle_id, 0) + 1
        except Exception:
            return json_str

        return json_str
