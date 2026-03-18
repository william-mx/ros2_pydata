from __future__ import annotations

import numpy as np

from std_msgs.msg import Header
from geometry_msgs.msg import Pose

from vision_msgs.msg import (
    Detection2D, Detection2DArray, BoundingBox2D, Pose2D, Point2D,
    Detection3D, Detection3DArray, BoundingBox3D,
    ObjectHypothesisWithPose, LabelInfo, VisionClass,
)

from ._utils import get_ros_timestamp, get_timestamp_unix
from .geometry_msgs import np_to_pose

from dataclasses import dataclass
from typing import Optional

# --- LabelInfo ---

def from_label_info(label_info_msg: LabelInfo) -> dict:
    """
    Converts a vision_msgs/LabelInfo message to a Python dict.

    Returns:
        dict: Mapping from class_id (int) to class_name (str).
    """
    return {vc.class_id: vc.class_name for vc in label_info_msg.class_map}


def to_label_info(id_to_label: dict, timestamp=None, frame_id='base_link') -> LabelInfo:
    """
    Converts an int-to-label dict to a vision_msgs/LabelInfo message.

    Args:
        id_to_label: Mapping from integer class IDs to label names.
        timestamp: Optional ROS timestamp or Unix float.
        frame_id: Header frame ID.

    Returns:
        vision_msgs.msg.LabelInfo
    """
    msg = LabelInfo()
    msg.header = Header()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.header.frame_id = frame_id
    msg.class_map = []
    for class_id, label in id_to_label.items():
        vc = VisionClass()
        vc.class_id = class_id
        vc.class_name = label
        msg.class_map.append(vc)
    return msg


# --- BoundingBox2D ---

def to_bbox2d(cx: float, cy: float, w: float, h: float, theta: float = 0.0) -> BoundingBox2D:
    """
    Creates a vision_msgs/BoundingBox2D from center coordinates and size.

    Args:
        cx, cy: Center coordinates.
        w, h: Width and height.
        theta: Rotation in radians (default: 0.0).

    Returns:
        vision_msgs.msg.BoundingBox2D
    """
    bbox = BoundingBox2D()
    bbox.center = Pose2D()
    bbox.center.position = Point2D(x=cx, y=cy)
    bbox.center.theta = theta
    bbox.size_x = w
    bbox.size_y = h
    return bbox


# --- Detection2D ---

# ---------------------------------------------------------------------------
# Detection2D dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection2D:
    """
    A single 2D detection result, decoded from a vision_msgs/Detection2DArray.

    Coordinates (x, y, width, height) are in pixels unless the array was decoded
    with img_width/img_height, in which case they are normalised to [0, 1].

    Attributes
    ----------
    seq : int, optional
        Caller-supplied sequence number shared across all detections in the
        same array (e.g. a frame counter). ``None`` if not provided.
    timestamp : float
        Unix timestamp in seconds from the message header stamp.
    label : str
        Human-readable class label (e.g. "car", "person").
    score : float
        Detection confidence in [0.0, 1.0].
    x : float
        Bounding-box centre x coordinate.
    y : float
        Bounding-box centre y coordinate.
    width : float
        Bounding-box width.
    height : float
        Bounding-box height.
    """

    seq: Optional[int]
    timestamp: float
    label: str
    score: float
    x: float
    y: float
    width: float
    height: float

    # ------------------------------------------------------------------
    # Derived properties — computed lazily, no storage overhead
    # ------------------------------------------------------------------

    @property
    def area(self) -> float:
        """Bounding-box area (width × height)."""
        return self.width * self.height

    @property
    def x1(self) -> float:
        """Left edge of the bounding box."""
        return self.x - self.width / 2

    @property
    def y1(self) -> float:
        """Top edge of the bounding box."""
        return self.y - self.height / 2

    @property
    def x2(self) -> float:
        """Right edge of the bounding box."""
        return self.x + self.width / 2

    @property
    def y2(self) -> float:
        """Bottom edge of the bounding box."""
        return self.y + self.height / 2

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """(x1, y1, x2, y2) corner format."""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """(cx, cy, width, height) centre format — same as the raw fields."""
        return self.x, self.y, self.width, self.height


# ---------------------------------------------------------------------------
# from_detection2d / from_detection2d_array
# ---------------------------------------------------------------------------

def from_detection2d(
    det,
    seq:        Optional[int] = None,
    img_width:  Optional[int] = None,
    img_height: Optional[int] = None,
) -> Detection2D:
    """
    Converts a single vision_msgs/Detection2D message to a :class:`Detection2D`.

    Parameters
    ----------
    det : vision_msgs.msg.Detection2D
        The individual detection message to decode.
    seq : int, optional
        Caller-supplied sequence number (e.g. a frame counter).
        Pass ``None`` (default) if not needed.
    img_width : int, optional
        Image width in pixels. When provided together with *img_height*, all
        coordinate fields are normalised to [0, 1].
    img_height : int, optional
        Image height in pixels. See *img_width*.

    Returns
    -------
    Detection2D
    """
    if det.results:
        hyp   = det.results[0].hypothesis
        label = hyp.class_id   # string label, e.g. "car"
        score = float(hyp.score)
    else:
        label = ""
        score = 0.0

    cx = det.bbox.center.position.x
    cy = det.bbox.center.position.y
    w  = det.bbox.size_x
    h  = det.bbox.size_y

    if img_width is not None and img_height is not None:
        cx /= img_width
        cy /= img_height
        w  /= img_width
        h  /= img_height

    return Detection2D(
        seq=seq,
        timestamp=get_timestamp_unix(det),
        label=label,
        score=score,
        x=cx,
        y=cy,
        width=w,
        height=h,
    )


def from_detection2d_array(
    msg,
    seq:        Optional[int] = None,
    img_width:  Optional[int] = None,
    img_height: Optional[int] = None,
) -> list[Detection2D]:
    """
    Converts a vision_msgs/Detection2DArray to a list of :class:`Detection2D`.

    Each detection reads its own timestamp from its individual header via
    :func:`._utils.get_timestamp_unix`.

    Parameters
    ----------
    msg : vision_msgs.msg.Detection2DArray
        The ROS2 message to decode.
    seq : int, optional
        Caller-supplied sequence number shared by all detections in this array
        (e.g. a frame counter). Pass ``None`` (default) if not needed.
    img_width : int, optional
        Image width in pixels. When provided together with *img_height*, all
        coordinate fields (x, y, width, height) are normalised to [0, 1].
    img_height : int, optional
        Image height in pixels. See *img_width*.

    Returns
    -------
    list[Detection2D]
        One dataclass instance per detection, all sharing the same *seq*.
    """
    return [
        from_detection2d(det, seq=seq, img_width=img_width, img_height=img_height)
        for det in msg.detections
    ]


# ---------------------------------------------------------------------------
# to_detection2d / to_detection2d_array
# ---------------------------------------------------------------------------

def _make_bbox2d(cx: float, cy: float, w: float, h: float) -> BoundingBox2D:
    bbox = BoundingBox2D()
    bbox.center = Pose2D()
    bbox.center.position = Point2D(x=float(cx), y=float(cy))
    bbox.center.theta = 0.0
    bbox.size_x = float(w)
    bbox.size_y = float(h)
    return bbox


def to_detection2d(
    label:     str,
    score:     float,
    cx:        float,
    cy:        float,
    w:         float,
    h:         float,
    timestamp=None,
    frame_id:  str = "base_link",
) -> "vision_msgs.msg.Detection2D":
    """
    Creates a vision_msgs/Detection2D from a *string label*, score, and bbox.

    Parameters
    ----------
    label : str
        Human-readable class label (e.g. ``"car"``). Stored directly in
        ``hypothesis.class_id`` — **not** a numeric ID.
    score : float
        Detection confidence in [0.0, 1.0].
    cx, cy : float
        Bounding-box centre coordinates (pixels).
    w, h : float
        Bounding-box width and height (pixels).
    timestamp : rclpy.time.Time or float or None
        Optional ROS timestamp or Unix float. Uses current time if None.
    frame_id : str
        Header frame ID.

    Returns
    -------
    vision_msgs.msg.Detection2D
    """
    from vision_msgs.msg import Detection2D as RosDetection2D  # avoid name clash

    msg = RosDetection2D()
    msg.header = Header()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.header.frame_id = frame_id
    msg.bbox = _make_bbox2d(cx, cy, w, h)

    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = str(label)   # always a string label
    hyp.hypothesis.score = float(score)
    msg.results.append(hyp)

    return msg


def to_detection2d_array(
    detections: list[tuple],
    timestamp=None,
    frame_id:   str = "base_link",
) -> "vision_msgs.msg.Detection2DArray":
    """
    Converts a list of 2D detections to a vision_msgs/Detection2DArray.

    Parameters
    ----------
    detections : list of tuples
        Each entry: ``(label: str, score: float, cx, cy, w, h)``.
        *label* must be the human-readable class name, not an integer ID.
    timestamp : rclpy.time.Time or float or None
        Optional timestamp. Uses current time if None.
    frame_id : str
        Header frame ID.

    Returns
    -------
    vision_msgs.msg.Detection2DArray
    """
    from vision_msgs.msg import Detection2DArray as RosArray  # avoid name clash

    array_msg = RosArray()
    array_msg.header.stamp = get_ros_timestamp(timestamp)
    array_msg.header.frame_id = frame_id

    for label, score, cx, cy, w, h in detections:
        array_msg.detections.append(
            to_detection2d(label, score, cx, cy, w, h, timestamp, frame_id)
        )

    return array_msg


# --- Detection3D ---

def to_detection3d(class_id, score: float, x: float, y: float, z: float,
                   timestamp=None, frame_id='base_link') -> Detection3D:
    """
    Creates a vision_msgs/Detection3D from class_id, score, and 3D position.

    Args:
        class_id: Integer class ID.
        score: Confidence score [0.0, 1.0].
        x, y, z: 3D position of the detected object.
        timestamp: Optional ROS timestamp or Unix float.
        frame_id: Header frame ID.

    Returns:
        vision_msgs.msg.Detection3D
    """
    detection = Detection3D()
    detection.header = Header()
    detection.header.stamp = get_ros_timestamp(timestamp)
    detection.header.frame_id = frame_id

    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = str(class_id)
    hypothesis.hypothesis.score = float(score)

    pose_stamped = np_to_pose(np.array([x, y, z]), yaw_angle=0.0)
    hypothesis.pose.pose = pose_stamped.pose

    bbox = BoundingBox3D()
    bbox.center = pose_stamped.pose
    bbox.size.x = 1.0
    bbox.size.y = 1.0
    bbox.size.z = 1.0

    detection.results.append(hypothesis)
    detection.bbox = bbox
    return detection


def to_detection3d_array(detections: list, timestamp=None, frame_id='base_link') -> Detection3DArray:
    """
    Converts a list of 3D detections to a vision_msgs/Detection3DArray.

    Args:
        detections: List of [class_id, score, x, y, z] entries.
        timestamp: Optional ROS timestamp or Unix float.
        frame_id: Header frame ID.

    Returns:
        vision_msgs.msg.Detection3DArray
    """
    array_msg = Detection3DArray()
    array_msg.header = Header()
    array_msg.header.stamp = get_ros_timestamp(timestamp)
    array_msg.header.frame_id = frame_id

    for class_id, score, x, y, z in detections:
        array_msg.detections.append(
            to_detection3d(class_id, score, x, y, z, timestamp, frame_id)
        )
    return array_msg


def from_detection3d(detection: Detection3D) -> list:
    """
    Extracts [class_id, score, x, y, z] from a vision_msgs/Detection3D.

    Returns:
        list: [class_id (int), score (float), x, y, z]
    """
    if not detection.results:
        return [None, 0.0, 0.0, 0.0, 0.0]

    result = detection.results[0]
    pos = result.pose.pose.position
    return [int(result.hypothesis.class_id), result.hypothesis.score, pos.x, pos.y, pos.z]


def from_detection3d_array(msg: Detection3DArray):
    """
    Converts a vision_msgs/Detection3DArray to a list of [class_id, score, x, y, z] entries.

    Returns:
        Tuple[list of lists, float]: (detections, Unix timestamp)
    """
    return [from_detection3d(d) for d in msg.detections], get_timestamp_unix(msg)
