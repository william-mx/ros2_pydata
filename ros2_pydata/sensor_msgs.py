import struct

import cv2
import numpy as np

from sensor_msgs.msg import Image, CompressedImage, Imu, PointCloud2, MagneticField, PointField

from ._utils import get_ros_timestamp, get_timestamp_unix


# --- Image ---

def image_to_np(msg):
    """Converts a sensor_msgs/Image message to a NumPy array (H x W x 3, uint8)."""
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    return image, get_timestamp_unix(msg)


def np_to_image(image: np.ndarray, timestamp=None) -> Image:
    """
    Converts a NumPy image (BGR or Grayscale) to a sensor_msgs/Image message.

    Args:
        image: H x W (mono8), H x W x 1 (mono8), or H x W x 3 (bgr8) array.
        timestamp: Optional ROS timestamp or Unix float.

    Returns:
        sensor_msgs.msg.Image

    Raises:
        ValueError: If the image shape is not supported.
    """
    ros_image = Image()

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
        ros_image.encoding = 'mono8'
    elif image.ndim == 3:
        height, width, channels = image.shape
        if channels == 1:
            ros_image.encoding = 'mono8'
        elif channels == 3:
            ros_image.encoding = 'bgr8'
        else:
            raise ValueError(f"Unsupported number of channels: {channels}. Expected 1 or 3.")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}. Expected 2 or 3.")

    ros_image.height = height
    ros_image.width = width
    ros_image.step = width * channels * image.itemsize
    ros_image.data = image.tobytes()
    ros_image.header.stamp = get_ros_timestamp(timestamp)

    return ros_image


# --- CompressedImage ---

def compressedimage_to_np(msg):
    """Converts a sensor_msgs/CompressedImage message to a NumPy array via JPEG decode."""
    image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    return image, get_timestamp_unix(msg)


def np_to_compressedimage(image: np.ndarray, timestamp=None) -> CompressedImage:
    """
    Converts a NumPy image to a sensor_msgs/CompressedImage message.
    Supports BGR (JPEG) and Mono8/Masks (PNG).
    """
    ros_image = CompressedImage()
    ros_image.header.stamp = get_ros_timestamp(timestamp)
    
    # Check if image is single-channel (Mask/Gray) or multi-channel (BGR)
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Use PNG for lossless integer preservation (Masks/Indices)
        ros_image.format = "png"
        success, encoded_img = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    else:
        # Use JPEG for standard camera images (Lossy but small)
        ros_image.format = "jpeg"
        # Optional: add params for quality, e.g., [cv2.IMWRITE_JPEG_QUALITY, 80]
        success, encoded_img = cv2.imencode('.jpg', image)

    if not success:
        raise RuntimeError("Could not encode image for ROS message")

    ros_image.data = np.array(encoded_img).tobytes()
    return ros_image


# --- MultiArray ---

def multiarray_to_np(msg, dtype=np.float32):
    """
    Converts a std_msgs/MultiArray message (e.g. Float32MultiArray) to a flat NumPy array.

    Args:
        msg: A MultiArray message.
        dtype: NumPy dtype for the output array. Defaults to np.float32.

    Returns:
        Tuple[np.ndarray, float]: (flat array of shape (N,), Unix timestamp)
    """
    return np.array(msg.data, dtype=dtype), get_timestamp_unix(msg)


# --- Imu ---

def imu_to_np(msg):
    """
    Converts a sensor_msgs/Imu message to a NumPy array (excluding orientation).

    Returns:
        Tuple[np.ndarray, float]: (array of shape (6,) [ax, ay, az, wx, wy, wz], Unix timestamp)
    """
    a = msg.linear_acceleration
    w = msg.angular_velocity
    arr = np.array([a.x, a.y, a.z, w.x, w.y, w.z])
    return arr, get_timestamp_unix(msg)


def np_to_imu(array: np.ndarray, frame_id='base_link', timestamp=None) -> Imu:
    """
    Converts a NumPy array of shape (6,) to a sensor_msgs/Imu message (excluding orientation).

    Args:
        array: [ax, ay, az, wx, wy, wz]
        frame_id: Header frame ID.
        timestamp: Optional ROS timestamp or Unix float.

    Returns:
        sensor_msgs.msg.Imu
    """
    msg = Imu()
    msg.header.frame_id = frame_id
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.linear_acceleration.x = float(array[0])
    msg.linear_acceleration.y = float(array[1])
    msg.linear_acceleration.z = float(array[2])
    msg.angular_velocity.x = float(array[3])
    msg.angular_velocity.y = float(array[4])
    msg.angular_velocity.z = float(array[5])
    return msg


# --- PointCloud2 ---

def np_to_pointcloud(points: np.ndarray, frame_id='base_link', timestamp=None) -> PointCloud2:
    """
    Creates a sensor_msgs/PointCloud2 message from an (N, 2) or (N, 3) NumPy array.

    Args:
        points: Array of shape (N, 2) [x, y] or (N, 3) [x, y, z]. 2D points get z=0.
        frame_id: Header frame ID.
        timestamp: Optional ROS timestamp or Unix float.

    Returns:
        sensor_msgs.msg.PointCloud2

    Raises:
        ValueError: If points does not have 2 or 3 columns.
    """
    points = np.array(points)

    if points.shape[1] not in [2, 3]:
        raise ValueError("Points must have 2 or 3 columns: [x, y] or [x, y, z].")
    if points.shape[1] == 2:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))

    msg = PointCloud2()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.header.frame_id = frame_id
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.is_dense = True

    point_data = bytearray()
    for point in points:
        point_data.extend(struct.pack('fff', point[0], point[1], point[2]))

    msg.data = bytes(point_data)
    msg.row_step = msg.point_step * points.shape[0]
    msg.height = 1
    msg.width = points.shape[0]
    return msg


# --- LaserScan ---

def scan_to_np(msg) -> np.ndarray:
    """
    Converts a sensor_msgs/LaserScan to an (N, 3) NumPy array of [x, y, intensity].

    Filters out non-finite (inf, NaN) range values.

    Returns:
        Tuple[np.ndarray, float]: (array of shape (N, 3), Unix timestamp).
                                   Shape (0, 3) if no valid points exist.
    """
    ranges_np = np.array(msg.ranges, dtype=np.float32)
    intensities_np = np.array(msg.intensities, dtype=np.float32)
    angles_np = msg.angle_min + np.arange(len(ranges_np)) * msg.angle_increment

    valid_mask = np.isfinite(ranges_np)
    valid_ranges = ranges_np[valid_mask]
    valid_angles = angles_np[valid_mask]
    valid_intensities = intensities_np[valid_mask]

    if valid_ranges.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)

    return np.column_stack((x, y, valid_intensities)).astype(np.float32), get_timestamp_unix(msg)


# --- MagneticField ---

def magneticfield_to_np(msg):
    """
    Converts a sensor_msgs/MagneticField message to a NumPy array of shape (3,).

    Returns:
        Tuple[np.ndarray, float]: ([x, y, z] magnetic field values, Unix timestamp)
    """
    data = np.array([msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z])
    return data, get_timestamp_unix(msg)


def np_to_magneticfield(array: np.ndarray, frame_id="base_link", timestamp=None) -> MagneticField:
    """
    Converts a NumPy array of shape (3,) to a sensor_msgs/MagneticField message.

    Args:
        array: [x, y, z] magnetic field values.
        frame_id: Header frame ID.
        timestamp: Optional ROS timestamp or Unix float.

    Returns:
        sensor_msgs.msg.MagneticField

    Raises:
        TypeError: If array is not a NumPy array of shape (3,).
    """
    if not isinstance(array, np.ndarray) or array.shape != (3,):
        raise TypeError("Input array must be a NumPy array of shape (3,).")

    msg = MagneticField()
    msg.header.frame_id = frame_id
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.magnetic_field.x = float(array[0])
    msg.magnetic_field.y = float(array[1])
    msg.magnetic_field.z = float(array[2])
    return msg
