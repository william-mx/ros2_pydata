import math

import numpy as np

from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Pose, Quaternion, PointStamped, PoseStamped

from ._utils import get_ros_timestamp, get_timestamp_unix


# --- Quaternion / Yaw helpers ---

def quaternion_to_yaw(quaternion: Quaternion) -> float:
    """
    Converts a geometry_msgs/Quaternion to a yaw angle (rotation around Z-axis).

    Args:
        quaternion: geometry_msgs.msg.Quaternion

    Returns:
        float: Yaw angle in radians.
    """
    t3 = +2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    t4 = +1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(t3, t4)


def yaw_to_quaternion(yaw_angle: float) -> Quaternion:
    """
    Converts a yaw angle (rotation around Z-axis) to a geometry_msgs/Quaternion.

    A yaw of 0 means the vehicle points in the +X direction.

    Args:
        yaw_angle: Yaw in radians.

    Returns:
        geometry_msgs.msg.Quaternion
    """
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = np.sin(yaw_angle / 2.0)
    q.w = np.cos(yaw_angle / 2.0)
    return q


# --- Pose ---

def pose_to_np(msg: PoseStamped):
    """
    Converts a geometry_msgs/PoseStamped to a NumPy position array and yaw angle.

    Returns:
        Tuple[np.ndarray, float, float]: (position (3,), yaw in radians, Unix timestamp)
    """
    point = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    yaw_angle = quaternion_to_yaw(msg.pose.orientation)
    return point, yaw_angle, get_timestamp_unix(msg)


def np_to_pose(point: np.ndarray, yaw_angle: float,
               frame_id='base_link', timestamp=None) -> PoseStamped:
    """
    Converts a NumPy position array and yaw angle to a geometry_msgs/PoseStamped.

    Args:
        point: Array of shape (2,) or (3,) — [x, y] or [x, y, z].
        yaw_angle: Yaw in radians (rotation around Z-axis).
        frame_id: Header frame ID.
        timestamp: Optional ROS timestamp or Unix float.

    Returns:
        geometry_msgs.msg.PoseStamped
    """
    msg = PoseStamped()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(point[0])
    msg.pose.position.y = float(point[1])
    msg.pose.position.z = float(point[2]) if len(point) > 2 else 0.0
    msg.pose.orientation = yaw_to_quaternion(yaw_angle)
    return msg


# --- Path ---

def np_to_path(waypoints, frame_id='base_link', timestamp=None) -> Path:
    """
    Converts a list or NumPy array of [x, y, theta] waypoints to a nav_msgs/Path.

    Args:
        waypoints: Array of shape (N, 3) — each row is [x, y, theta].
                   A single waypoint may be passed as a 1D array of length 3.
        frame_id: Header frame ID.
        timestamp: Optional ROS timestamp or Unix float.

    Returns:
        nav_msgs.msg.Path

    Raises:
        ValueError: If waypoints do not have exactly 3 columns.
    """
    waypoints = np.array(waypoints)
    if waypoints.ndim == 1:
        waypoints = waypoints.reshape(1, -1)

    if waypoints.shape[1] != 3:
        raise ValueError("Waypoints must have 3 columns: [x, y, theta].")

    path_msg = Path()
    path_msg.header.stamp = get_ros_timestamp(timestamp)
    path_msg.header.frame_id = frame_id

    for wp in waypoints:
        pose = np_to_pose(wp[:2], wp[2])
        pose.header.frame_id = frame_id
        path_msg.poses.append(pose)

    return path_msg


# --- PointStamped ---

def np_to_point(point: np.ndarray, frame_id='base_link', timestamp=None) -> PointStamped:
    """
    Converts a NumPy array to a geometry_msgs/PointStamped message.

    Args:
        point: Array of shape (2,) or (3,) — [x, y] or [x, y, z].
        frame_id: Header frame ID.
        timestamp: Optional ROS timestamp or Unix float.

    Returns:
        geometry_msgs.msg.PointStamped
    """
    msg = PointStamped()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.header.frame_id = frame_id
    msg.point.x = float(point[0])
    msg.point.y = float(point[1])
    msg.point.z = float(point[2]) if len(point) > 2 else 0.0
    return msg


def point_to_np(msg: PointStamped):
    """
    Converts a geometry_msgs/PointStamped to a NumPy array and Unix timestamp.

    Returns:
        Tuple[np.ndarray, float]: (array of shape (3,), Unix timestamp)
    """
    point = np.array([msg.point.x, msg.point.y, msg.point.z])
    return point, get_timestamp_unix(msg)
