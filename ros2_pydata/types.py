from dataclasses import dataclass

from typing import Optional, Tuple

# --- Detection2D dataclass ---

@dataclass
class Detection2DResult:
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

    @property
    def area(self) -> float:
        """Bounding-box area (width × height)."""
        return self.width * self.height

    @property
    def x1(self) -> float:
        return self.x - self.width / 2

    @property
    def y1(self) -> float:
        return self.y - self.height / 2

    @property
    def x2(self) -> float:
        return self.x + self.width / 2

    @property
    def y2(self) -> float:
        return self.y + self.height / 2

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height
    

# --- Detection3D dataclass ---

@dataclass
class Detection3DResult:
    """
    A single 3D detection result, decoded from a vision_msgs/Detection3DArray.

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
    x, y, z : float
        3D position of the bounding box center.
    rx, ry, rz, rw : float
        Orientation quaternion of the bounding box.
    sx, sy, sz : float
        Bounding box size along each axis.
    id : str
        Consistency ID across detection messages (empty string if not set).
    """

    seq:       Optional[int]
    timestamp: float
    label:     str
    score:     float
    x:         float
    y:         float
    z:         float
    rx:        float
    ry:        float
    rz:        float
    rw:        float
    sx:        float
    sy:        float
    sz:        float
    id:        str = ""

    @property
    def position(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def orientation(self) -> Tuple[float, float, float, float]:
        """Quaternion as (rx, ry, rz, rw)."""
        return self.rx, self.ry, self.rz, self.rw

    @property
    def size(self) -> Tuple[float, float, float]:
        return self.sx, self.sy, self.sz
