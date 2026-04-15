"""Core data structures for scene representation.

Defines Object, SpatialRelation, and SceneGraph — the structured
output of Stage 1 perception. Every downstream component operates on
these types rather than raw pixel arrays.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Object:
    """A segmented object in the scene.

    Attributes:
        obj_id: Persistent ID assigned by the tracker across frames.
        color: Color index (0-15).
        pixels: Set of (y, x) pixel coordinates belonging to this object.
        bbox: Bounding box as (min_y, min_x, max_y, max_x).
        centroid: Float centroid as (cy, cx).
        area: Pixel count.
        confidence: Segmentation confidence (1.0 = certain).
    """

    obj_id: int
    color: int
    pixels: set  # set of (y, x) tuples
    bbox: tuple  # (min_y, min_x, max_y, max_x)
    centroid: tuple  # (cy, cx) floats
    area: int
    confidence: float = 1.0

    @property
    def width(self) -> int:
        """Width of the bounding box in pixels."""
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        """Height of the bounding box in pixels."""
        return self.bbox[2] - self.bbox[0] + 1


@dataclass
class SpatialRelation:
    """Spatial relationship between two objects.

    Attributes:
        obj_a_id: ID of the first object.
        obj_b_id: ID of the second object.
        relation: One of 'above', 'below', 'left_of', 'right_of',
            'adjacent', 'inside', 'overlapping'.
        distance: Manhattan distance between centroids (0 for adjacent).
    """

    obj_a_id: int
    obj_b_id: int
    relation: str
    distance: float


@dataclass
class SceneGraph:
    """Complete structured representation of a single frame.

    Attributes:
        objects: Mapping from obj_id to Object.
        relations: All pairwise SpatialRelations between objects.
        background_color: Most-frequent color, treated as background.
        frame_hash: Hash of raw frame bytes for fast equality checks.
        raw_frame: Original numpy array kept for fallback operations.
    """

    objects: dict  # obj_id -> Object
    relations: list  # list of SpatialRelation
    background_color: int
    frame_hash: int
    raw_frame: object  # numpy array

    def object_count(self) -> int:
        """Return the number of objects in this scene."""
        return len(self.objects)

    def objects_by_color(self, color: int) -> list:
        """Return all objects of a given color.

        Args:
            color: Color index to filter by.

        Returns:
            List of Object instances with that color.
        """
        return [o for o in self.objects.values() if o.color == color]

    def get_object_at(self, y: int, x: int) -> Optional[Object]:
        """Return the object containing pixel (y, x), or None.

        Args:
            y: Row coordinate.
            x: Column coordinate.

        Returns:
            Object if found, else None.
        """
        for obj in self.objects.values():
            if (y, x) in obj.pixels:
                return obj
        return None
