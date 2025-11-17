from google.transit import gtfs_realtime_pb2
import base64
import json


def decode_gtfs_feed(data_string):
    """
    Decodes a base64-encoded GTFS Realtime FeedMessage protobuf string 
    into individual vehicle position dictionaries (JSON strings).

    This function is designed to be used with PyFlink's flat_map:
    * It handles base64 decoding and protobuf parsing.
    * It iterates through each vehicle entity in the feed.
    * It extracts key vehicle data (ID, trip/route ID, lat/lon, timestamp).
    * It serializes each vehicle's data into a JSON string and yields it.
    * It gracefully handles errors in the input data or parsing process.

    Args:
        data_string (str): The raw data string, expected to be a base64-
                           encoded GTFS-RT FeedMessage binary.

    Yields:
        str: A JSON string representing a single vehicle's position data, 
             or an "Error" string if parsing fails.
    """
    try:
        if data_string.startswith("Error"):
            yield data_string
            return

        binary_data = base64.b64decode(data_string)

        # Parse the protobuf
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(binary_data)

        # Process each vehicle entity
        for entity in feed.entity:
            if entity.HasField('vehicle'):
                vehicle = entity.vehicle
                vehicle_data = {
                    'vehicle_id': vehicle.vehicle.id if vehicle.HasField('vehicle') else None,
                    'trip_id': vehicle.trip.trip_id if vehicle.HasField('trip') else None,
                    'route_id': vehicle.trip.route_id if vehicle.HasField('trip') else None,
                    'latitude': vehicle.position.latitude if vehicle.HasField('position') else None,
                    'longitude': vehicle.position.longitude if vehicle.HasField('position') else None,
                    'timestamp': vehicle.timestamp if vehicle.HasField('timestamp') else None
                }
                yield json.dumps(vehicle_data)

    except Exception as e:
        yield f"Error: {str(e)}"
