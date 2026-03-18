import time
import warnings

import rclpy
from rclpy.clock import Clock

from builtin_interfaces.msg import Time as TimeMsg


def get_ros_timestamp(timestamp=None) -> TimeMsg:
    """
    Ensures a ROS timestamp message (builtin_interfaces.msg.Time) is returned.

    Args:
        timestamp: Can be None, rclpy.time.Time, builtin_interfaces.msg.Time, or float (Unix seconds).

    Returns:
        A builtin_interfaces.msg.Time message suitable for msg.header.stamp.
    """
    if isinstance(timestamp, rclpy.time.Time):
        return timestamp.to_msg()
    elif isinstance(timestamp, TimeMsg):
        return timestamp
    elif isinstance(timestamp, float):
        sec = int(timestamp)
        nanosec = int((timestamp - sec) * 1e9)
        return TimeMsg(sec=sec, nanosec=nanosec)
    else:
        return Clock().now().to_msg()


def get_timestamp_unix(msg) -> float:
    """
    Extracts the Unix timestamp (seconds since epoch) from a message header's stamp.

    Args:
        msg: A ROS message with a header.stamp attribute.

    Returns:
        float: Unix timestamp in seconds, or current system time if header is missing.
    """
    try:
        stamp = msg.header.stamp
        return stamp.sec + 1e-9 * stamp.nanosec
    except AttributeError:
        warnings.warn("Message does not have a valid header.stamp. Returning current system time.")
        return time.time()
