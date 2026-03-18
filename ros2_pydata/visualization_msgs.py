import numpy as np

from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import ImageMarker

from ._utils import get_ros_timestamp


def pc_to_image_marker(points_2d: np.ndarray, frame_id: str = "base_link",
                       timestamp=None, depth: np.ndarray = None,
                       scale: float = 2.0) -> ImageMarker:
    """
    Converts an (N, 2) NumPy array of image coordinates to a visualization_msgs/ImageMarker.

    The marker type is POINTS. Optionally colorizes each point by depth using a
    blue (near) to red (far) colormap.

    Args:
        points_2d: (N, 2) array of [x, y] image pixel coordinates.
        frame_id: Header frame ID (e.g. 'camera_frame').
        timestamp: Optional ROS timestamp or Unix float.
        depth: Optional (N,) array of depth values used to colorize points.
        scale: Point diameter in pixels. Default is 2.0.

    Returns:
        visualization_msgs.msg.ImageMarker
    """
    marker = ImageMarker()
    marker.header = Header()
    marker.header.stamp = get_ros_timestamp(timestamp)
    marker.header.frame_id = frame_id
    marker.type = ImageMarker.POINTS
    marker.scale = scale

    for i, (x, y) in enumerate(points_2d):
        marker.points.append(Point(x=float(x), y=float(y), z=0.0))

        if depth is not None:
            val = float(depth[i])
            d_norm = np.clip((val - np.min(depth)) / (np.ptp(depth) + 1e-5), 0.0, 1.0)
            marker.outline_colors.append(ColorRGBA(r=1 - d_norm, g=0.3, b=d_norm, a=0.9))

    if not marker.outline_colors:
        marker.outline_color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.8)

    return marker
