"""Perception package: Stage 1 — object segmentation and tracking."""

from .scene_graph import Object, SpatialRelation, SceneGraph
from .segmentation import segment_frame
from .tracker import ObjectTracker

__all__ = ["Object", "SpatialRelation", "SceneGraph", "segment_frame", "ObjectTracker"]
