from pyflink.datastream.functions import MapFunction
from collections import defaultdict
from utils import haversine_distance_km
import json
from collections import deque


class AppendSpeedToStream(MapFunction):
    def __init__(self):
        # history per vehicle
        self.last_points = defaultdict(lambda: deque(maxlen=1))

    def map(self, json_str: str) -> str:
        vehicle = json.loads(json_str)
        vid = vehicle.get("vehicle_id")
        lat = vehicle.get("latitude")
        lon = vehicle.get("longitude")
        ts = vehicle.get("timestamp")

        if vid is None or lat is None or lon is None or ts is None:
            vehicle["speed_kmh"] = 0.0
            return json.dumps(vehicle)

        history = self.last_points[vid]

        if len(history) == 0:
            speed = 0.0
        else:
            prev_lat, prev_lon, prev_ts = history[0]

            dt = ts - prev_ts
            if dt <= 0:
                speed = 0.0
            else:
                dist = haversine_distance_km(prev_lat, prev_lon, lat, lon)
                speed = (dist / dt) * 3600.0

        # update history
        history.clear()
        history.append((lat, lon, ts))

        vehicle["speed_kmh"] = speed
        return json.dumps(vehicle)
