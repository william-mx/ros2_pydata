from ackermann_msgs.msg import AckermannDriveStamped

from ._utils import get_ros_timestamp, get_timestamp_unix


def from_ackermann(msg: AckermannDriveStamped):
    """
    Extracts speed, steering angle, and Unix timestamp from an AckermannDriveStamped message.

    Returns:
        Tuple[float, float, float]: (speed in m/s, steering angle in radians, Unix timestamp)
    """
    return msg.drive.speed, msg.drive.steering_angle, get_timestamp_unix(msg)


def to_ackermann(speed: float, steering_angle: float, timestamp=None) -> AckermannDriveStamped:
    """
    Creates an AckermannDriveStamped message from speed and steering angle.

    Args:
        speed: Longitudinal speed in m/s.
        steering_angle: Steering angle in radians.
        timestamp: Optional ROS timestamp or Unix float.

    Returns:
        ackermann_msgs.msg.AckermannDriveStamped
    """
    msg = AckermannDriveStamped()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.drive.speed = float(speed)
    msg.drive.steering_angle = float(steering_angle)
    return msg
