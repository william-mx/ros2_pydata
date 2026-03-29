# ros2_pydata/__init__.py

# Always available — no ROS required
from .types import Detection2DResult, Detection3DResult

# ROS-only — only available when rclpy is installed
try:
    from ._utils import get_ros_timestamp, get_timestamp_unix

    from .sensor_msgs import (
        image_to_np,
        np_to_image,
        compressedimage_to_np,
        np_to_compressedimage,
        multiarray_to_np,
        imu_to_np,
        np_to_imu,
        np_to_pointcloud,
        scan_to_np,
        magneticfield_to_np,
        np_to_magneticfield,
    )

    from .geometry_msgs import (
        quaternion_to_yaw,
        yaw_to_quaternion,
        pose_to_np,
        np_to_pose,
        np_to_path,
        np_to_point,
        point_to_np,
    )

    from .vision_msgs import (
        from_label_info,
        to_label_info,
        to_detection2d,
        to_detection2d_array,
        from_detection2d,
        from_detection2d_array,
        to_detection3d,
        to_detection3d_array,
        from_detection3d,
        from_detection3d_array,
    )

    from .ackermann_msgs import (
        from_ackermann,
        to_ackermann,
    )

    from .visualization_msgs import (
        pc_to_image_marker,
    )

except ModuleNotFoundError:
    pass  # ROS not available — only types are accessible