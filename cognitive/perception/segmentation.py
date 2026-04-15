"""Object segmentation from raw 64x64 grid frames.

Uses flood-fill (BFS) connected-component labeling to extract objects.
Background is identified as the most frequent color; all other
contiguous same-color regions of sufficient size become Objects.
"""

import numpy as np
from collections import deque

from .scene_graph import Object, SpatialRelation, SceneGraph


def segment_frame(frame: np.ndarray, min_size: int = 2) -> SceneGraph:
    """Segment a 64x64 grid into objects via flood-fill.

    Detects the background color (most frequent), then labels every
    contiguous foreground region as a distinct Object. Runs in O(H*W)
    which is well within the <10ms budget for 64x64 grids.

    Args:
        frame: 64x64 numpy array of int (color indices 0-15).
        min_size: Minimum pixel count for a region to be kept as an object.
            Lower values surface small details; higher values reduce noise.

    Returns:
        SceneGraph containing all detected objects and their spatial relations.
    """
    frame = np.asarray(frame, dtype=np.int32)
    h, w = frame.shape

    counts = np.bincount(frame.flatten(), minlength=16)
    bg_color = int(np.argmax(counts))

    visited = np.zeros((h, w), dtype=bool)
    objects = {}
    obj_id = 0

    for y in range(h):
        for x in range(w):
            if visited[y, x] or frame[y, x] == bg_color:
                visited[y, x] = True
                continue

            color = frame[y, x]
            pixels: set = set()
            queue = deque([(y, x)])
            visited[y, x] = True

            while queue:
                cy, cx = queue.popleft()
                pixels.add((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < h and 0 <= nx < w
                            and not visited[ny, nx]
                            and frame[ny, nx] == color):
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(pixels) < min_size:
                continue

            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            obj = Object(
                obj_id=obj_id,
                color=int(color),
                pixels=pixels,
                bbox=(min(ys), min(xs), max(ys), max(xs)),
                centroid=(float(np.mean(ys)), float(np.mean(xs))),
                area=len(pixels),
            )
            objects[obj_id] = obj
            obj_id += 1

    relations = _extract_relations(objects)

    return SceneGraph(
        objects=objects,
        relations=relations,
        background_color=bg_color,
        frame_hash=hash(frame.tobytes()),
        raw_frame=frame.copy(),
    )


def _extract_relations(objects: dict) -> list:
    """Compute pairwise spatial relations between all object pairs.

    Args:
        objects: Mapping from obj_id to Object.

    Returns:
        List of SpatialRelation instances.
    """
    relations = []
    obj_list = list(objects.values())

    for i, a in enumerate(obj_list):
        for b in obj_list[i + 1:]:
            dy = b.centroid[0] - a.centroid[0]
            dx = b.centroid[1] - a.centroid[1]
            dist = abs(dy) + abs(dx)

            primary = (
                ('below' if dy > 0 else 'above') if abs(dy) >= abs(dx)
                else ('right_of' if dx > 0 else 'left_of')
            )
            relations.append(SpatialRelation(a.obj_id, b.obj_id, primary, dist))

            adjacent = False
            for py, px in a.pixels:
                for ddy, ddx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (py + ddy, px + ddx) in b.pixels:
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent:
                relations.append(
                    SpatialRelation(a.obj_id, b.obj_id, 'adjacent', 0.0)
                )

    return relations
