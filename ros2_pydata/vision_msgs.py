from __future__ import annotations

import numpy as np

from std_msgs.msg import Header
from geometry_msgs.msg import Pose

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    BoundingBox2D, BoundingBox3D, Pose2D, Point2D,
    ObjectHypothesisWithPose, LabelInfo, VisionClass,
)

from ._utils import get_ros_timestamp, get_timestamp_unix
from .geometry_msgs import np_to_pose
from .types import Detection2DResult, Detection3DResult

from typing import Optional, List, Tuple

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
    msg.class_map = [
        VisionClass(class_id=class_id, class_name=label)
        for class_id, label in id_to_label.items()
    ]
    return msg


# --- from_detection2d / from_detection2d_array ---

def from_detection2d(
    det:        Detection2D,
    seq:        Optional[int] = None,
    img_width:  Optional[int] = None,
    img_height: Optional[int] = None,
) -> Detection2DResult:
    """
    Converts a single vision_msgs/Detection2D message to a :class:`Detection2D`.

    Parameters
    ----------
    det : vision_msgs.msg.Detection2D
    seq : int, optional
    img_width, img_height : int, optional
        When both provided, coordinates are normalised to [0, 1].

    Returns
    -------
    Detection2D
    """
    if det.results:
        label = det.results[0].hypothesis.class_id
        score = float(det.results[0].hypothesis.score)
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

    return Detection2DResult(
        seq=seq,
        timestamp=get_timestamp_unix(det),
        label=label,
        score=score,
        x=cx, y=cy, width=w, height=h,
    )


def from_detection2d_array(
    msg,
    seq:        Optional[int] = None,
    img_width:  Optional[int] = None,
    img_height: Optional[int] = None,
) -> List[Detection2DResult]:
    """
    Converts a vision_msgs/Detection2DArray to a List of :class:`Detection2D`.
    """
    return [
        from_detection2d(det, seq=seq, img_width=img_width, img_height=img_height)
        for det in msg.detections
    ]


# --- to_detection2d / to_detection2d_array ---

def _make_bbox2d(cx: float, cy: float, w: float, h: float) -> BoundingBox2D:
    bbox = BoundingBox2D()
    bbox.center = Pose2D()
    bbox.center.position = Point2D(x=float(cx), y=float(cy))
    bbox.center.theta = 0.0
    bbox.size_x = float(w)
    bbox.size_y = float(h)
    return bbox


def to_detection2d(
    label:    str,
    score:    float,
    cx:       float,
    cy:       float,
    w:        float,
    h:        float,
    timestamp=None,
    frame_id: str = "base_link",
) -> Detection2D:
    """
    Creates a vision_msgs/Detection2D from a string label, score, and bbox.
    """
    msg = Detection2D()
    msg.header = Header()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.header.frame_id = frame_id
    msg.bbox = _make_bbox2d(cx, cy, w, h)

    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = str(label)
    hyp.hypothesis.score = float(score)
    msg.results.append(hyp)

    return msg


def to_detection2d_array(
    detections: List[Tuple],
    timestamp=None,
    frame_id:   str = "base_link",
) -> Detection2DArray:
    """
    Converts a list of ``(label, score, cx, cy, w, h)`` tuples to a
    vision_msgs/Detection2DArray.
    """
    msg = Detection2DArray()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.header.frame_id = frame_id
    for label, score, cx, cy, w, h, *_ in detections:
        msg.detections.append(to_detection2d(label, score, cx, cy, w, h, timestamp, frame_id))
    return msg


# --- from_detection3d / from_detection3d_array ---

def from_detection3d(
    det: Detection3D,
    seq: Optional[int] = None,
) -> Detection3DResult:
    """
    Converts a single vision_msgs/Detection3D message to a :class:`Detection3D`.
    """
    if det.results:
        label = det.results[0].hypothesis.class_id
        score = float(det.results[0].hypothesis.score)
    else:
        label = ""
        score = 0.0

    pos  = det.bbox.center.position
    ori  = det.bbox.center.orientation
    size = det.bbox.size

    return Detection3DResult(
        seq=seq,
        timestamp=get_timestamp_unix(det),
        label=label,
        score=score,
        x=pos.x, y=pos.y, z=pos.z,
        rx=ori.x, ry=ori.y, rz=ori.z, rw=ori.w,
        sx=size.x, sy=size.y, sz=size.z,
        id=det.id,
    )


def from_detection3d_array(
    msg,
    seq: Optional[int] = None,
) -> Tuple[List[Detection3DResult], float]:
    """
    Converts a vision_msgs/Detection3DArray to a List of :class:`Detection3D`.

    Returns
    -------
    Tuple[List[Detection3DResult], float]
        Detections and Unix timestamp from the array header.
    """
    return (
        [from_detection3d(d, seq=seq) for d in msg.detections],
        get_timestamp_unix(msg),
    )


# --- to_detection3d / to_detection3d_array ---

def to_detection3d(
    class_id,
    score:    float,
    x:        float,
    y:        float,
    z:        float,
    timestamp=None,
    frame_id: str = 'base_link',
) -> Detection3D:
    """
    Creates a vision_msgs/Detection3D from class_id, score, and 3D position.
    """
    msg = Detection3D()
    msg.header = Header()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.header.frame_id = frame_id

    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = str(class_id)
    hyp.hypothesis.score = float(score)

    pose_stamped = np_to_pose(np.array([x, y, z]), yaw_angle=0.0)
    hyp.pose.pose = pose_stamped.pose

    bbox = BoundingBox3D()
    bbox.center = pose_stamped.pose
    bbox.size.x = bbox.size.y = bbox.size.z = 1.0

    msg.results.append(hyp)
    msg.bbox = bbox
    return msg


def to_detection3d_array(
    detections: List,
    timestamp=None,
    frame_id: str = 'base_link',
) -> Detection3DArray:
    """
    Converts a list of ``[class_id, score, x, y, z]`` entries to a
    vision_msgs/Detection3DArray.
    """
    msg = Detection3DArray()
    msg.header = Header()
    msg.header.stamp = get_ros_timestamp(timestamp)
    msg.header.frame_id = frame_id
    for class_id, score, x, y, z in detections:
        msg.detections.append(to_detection3d(class_id, score, x, y, z, timestamp, frame_id))
    return msg