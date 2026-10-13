"""CV-based map parser: road skeleton extraction + junction (intersection) detection.

No neural network needed. Pure computer vision:

    map.png (RGB)
      → extract_semantic_channels (reuse existing color-threshold: water/road/building/green)
      → road_mask (binary)
      → skeletonize (Zhang-Suen thinning, pure numpy — no skimage dependency)
      → build graph (nodes = junction pixels, edges = road segments)
      → node_degree analysis: degree >= 3 = intersection/junction
      → output: intersection_mask, road_graph, junction_positions

This gives the model explicit topological structure (where intersections are)
that the current 16-token VAE path completely lacks.
"""
import math
import numpy as np
from PIL import Image
import cv2
from typing import Optional


# ---------------------------------------------------------------------------
# Semantic extraction (compatible with map_latent.extract_semantic_map)
# ---------------------------------------------------------------------------

def extract_semantic_channels(map_path: str, target_hw: Optional[tuple] = None) -> np.ndarray:
    """Extract 4-channel semantic map: [H, W, 4] (water/road/building/green).

    Compatible with map_latent.extract_semantic_map but returns unnormalized
    raw channels (each in [0,1]) for cleaner downstream processing.
    """
    img = Image.open(map_path).convert("RGB")
    if target_hw is not None:
        img = img.resize((target_hw[1], target_hw[0]), Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    water = np.clip((b - np.maximum(r, g)) * 2.0, 0, 1)
    water *= np.clip(b, 0.1, 1.0)

    lightness = (r + g + b) / 3.0
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = max_c - min_c
    road = np.clip((lightness - 0.3) * 3.0, 0, 1) * np.clip(1.0 - sat * 3.0, 0, 1)

    green = np.clip((g - np.maximum(r, b)) * 2.5, 0, 1)
    building = np.clip(1.0 - water - road - green, 0, 1)

    return np.stack([water, road, building, green], axis=-1)


# ---------------------------------------------------------------------------
# Road mask + skeletonization
# ---------------------------------------------------------------------------

def extract_road_mask(map_path: str, target_hw: Optional[tuple] = None,
                      threshold: float = 0.3) -> np.ndarray:
    """Extract binary road mask from map image.

    Returns: [H, W] uint8, 255=road, 0=non-road.
    """
    sem = extract_semantic_channels(map_path, target_hw)
    road = sem[..., 1]  # road channel
    mask = (road > threshold).astype(np.uint8) * 255
    # Clean up: remove small noise, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen skeletonization (pure numpy, no skimage dependency).

    Input: binary [H, W] with 1=foreground, 0=background.
    Returns: skeleton [H, W] with 1=skeleton, 0=background.
    """
    skel = binary.copy()
    changing = True
    while changing:
        # Step 1
        neighbors = _count_neighbors(skel)
        transitions = _count_transitions(skel)
        # Condition A: 2 <= P1 <= 6
        cond_A = (neighbors >= 2) & (neighbors <= 6)
        # Condition B: transitions == 1
        cond_B = (transitions == 1)
        # Condition C: P2*P4*P6 == 0
        # Condition D: P4*P6*P8 == 0
        H, W = skel.shape
        p2 = np.roll(skel, -1, axis=0)  # above
        p4 = np.roll(skel, 1, axis=1)   # right
        p6 = np.roll(skel, 1, axis=0)   # below
        p8 = np.roll(skel, -1, axis=1)  # left
        cond_C = (p2 * p4 * p6) == 0
        cond_D = (p4 * p6 * p8) == 0
        to_remove = cond_A & cond_B & cond_C & cond_D & (skel == 1)
        changed1 = to_remove.any()
        skel[to_remove] = 0

        # Step 2
        neighbors = _count_neighbors(skel)
        transitions = _count_transitions(skel)
        cond_A = (neighbors >= 2) & (neighbors <= 6)
        cond_B = (transitions == 1)
        # Condition C': P2*P4*P8 == 0
        # Condition D': P2*P6*P8 == 0
        cond_C2 = (p2 * p4 * p8) == 0
        cond_D2 = (p2 * p6 * p8) == 0
        to_remove = cond_A & cond_B & cond_C2 & cond_D2 & (skel == 1)
        changed2 = to_remove.any()
        skel[to_remove] = 0

        changing = changed1 or changed2
    return skel


def _count_neighbors(skel: np.ndarray) -> np.ndarray:
    """Count 8-connected neighbors for each pixel."""
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, kernel).astype(np.int32)


def _count_transitions(skel: np.ndarray) -> np.ndarray:
    """Count 0→1 transitions in clockwise order around each pixel."""
    H, W = skel.shape
    s = skel.astype(np.int32)
    p2 = np.roll(s, -1, axis=0)
    p3 = np.roll(np.roll(s, -1, axis=0), 1, axis=1)
    p4 = np.roll(s, 1, axis=1)
    p5 = np.roll(np.roll(s, 1, axis=0), 1, axis=1)
    p6 = np.roll(s, 1, axis=0)
    p7 = np.roll(np.roll(s, 1, axis=0), -1, axis=1)
    p8 = np.roll(s, -1, axis=1)
    p9 = np.roll(np.roll(s, -1, axis=0), -1, axis=1)
    seq = [p2, p3, p4, p5, p6, p7, p8, p9]
    transitions = np.zeros_like(s)
    for i in range(8):
        transitions += (seq[i] == 0) & (seq[(i + 1) % 8] == 1)
    return transitions


def skeletonize_road(road_mask: np.ndarray, min_road_width: int = 3) -> np.ndarray:
    """Skeletonize a road mask to a 1-pixel-wide centerline.

    Uses morphological thinning (Zhang-Suen) after optional pre-thinning
    to reduce wide roads. Returns binary skeleton [H, W] uint8 (0 or 255).

    For wide roads, first erode to reduce width, then thin.
    """
    binary = (road_mask > 127).astype(np.uint8)
    # For wide roads, pre-erode to get a cleaner skeleton
    if min_road_width > 1:
        kernel_size = max(1, min_road_width - 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.erode(binary, kernel, iterations=1)

    # Use OpenCV's ximgproc thinning if available, otherwise Zhang-Suen
    try:
        thin = cv2.ximgproc.thinning(binary * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return thin
    except (AttributeError, cv2.error):
        # Fallback: Zhang-Suen in numpy
        skel = _zhang_suen_thinning(binary)
        return (skel * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Junction detection via distance transform (robust, no skeleton needed)
# ---------------------------------------------------------------------------

def detect_junctions(road_mask: np.ndarray, dilation: int = 15,
                     min_area: int = 20, width_ratio: float = 1.8,
                     width_offset: float = 3.0) -> tuple:
    """Detect road junctions/intersections from a road mask.

    Uses distance transform: at junctions, multiple roads converge so the
    road is wider there. Pixels where the distance-to-background exceeds
    ~1.8x the median road half-width are junction candidates. Connected
    components of these wide-road regions are individual junctions.

    This is far more robust than skeleton-based degree counting, which
    produces hundreds of false positives from skeleton noise.

    Args:
        road_mask: binary road mask [H, W] uint8 (0 or 255).
        dilation: radius to dilate junction regions in the output mask.
        min_area: minimum pixel area for a junction component.
        width_ratio: junction width threshold = median_width * width_ratio.
        width_offset: additional offset on the width threshold.

    Returns:
        (junction_mask, junction_points)
        - junction_mask: [H, W] uint8, 255 at junction areas, 0 elsewhere.
        - junction_points: list of (x, y, area) for each junction, sorted
          by area (largest/most-significant first).
    """
    binary = (road_mask > 127).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    road_pixels = dist[binary > 0]
    if len(road_pixels) == 0:
        return np.zeros_like(road_mask, dtype=np.uint8), []

    median_width = float(np.median(road_pixels))
    junction_threshold = max(median_width * width_ratio, median_width + width_offset)

    # Wide road = junction candidates
    wide_road = (dist > junction_threshold).astype(np.uint8) * 255

    # Connected components = individual junctions
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wide_road, connectivity=8)

    junction_points = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        junction_points.append((cx, cy, area))

    # Sort by area (largest = most significant intersection first)
    junction_points.sort(key=lambda x: -x[2])

    # Build junction mask (dilate each junction region)
    junction_mask = np.zeros_like(road_mask, dtype=np.uint8)
    # Use the wide_road components directly, then dilate
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2, dilation * 2))
        junction_mask = cv2.bitwise_or(junction_mask, cv2.dilate(comp, kernel))

    return junction_mask, junction_points


# ---------------------------------------------------------------------------
# Route mask rasterization
# ---------------------------------------------------------------------------

def rasterize_route(polyline: np.ndarray, shape: tuple, thickness: int = 3) -> np.ndarray:
    """Rasterize a route polyline into a binary mask.

    Args:
        polyline: [N, 2] array of (x, y) pixel coordinates.
        shape: (H, W) of the output mask.
        thickness: line thickness in pixels.

    Returns: [H, W] uint8, 255=route, 0=background.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    pts = polyline.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], False, 255, thickness)
    return mask


def route_progress_segment(polyline: np.ndarray, progress_start: float,
                           progress_end: float) -> np.ndarray:
    """Extract the segment of a polyline between two progress values [0,1].

    Uses cumulative arc length for parameterization.
    Returns: [M, 2] sub-polyline from progress_start to progress_end.
    """
    diffs = np.diff(polyline, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seglens)])
    total = cum[-1]
    if total < 1e-9:
        return polyline[:1].copy()

    cum_norm = cum / total
    # Find start and end points by interpolation
    s_start = progress_start
    s_end = min(progress_end, 1.0)

    result = []
    # Start point
    if s_start <= 0:
        result.append(polyline[0].copy())
    else:
        seg_idx = np.searchsorted(cum_norm, s_start) - 1
        seg_idx = max(0, min(len(polyline) - 2, seg_idx))
        t = (s_start - cum_norm[seg_idx]) / (cum_norm[seg_idx + 1] - cum_norm[seg_idx] + 1e-9)
        pt = polyline[seg_idx] + t * (polyline[seg_idx + 1] - polyline[seg_idx])
        result.append(pt)

    # Intermediate points
    for i in range(len(polyline)):
        if s_start < cum_norm[i] < s_end:
            result.append(polyline[i].copy())

    # End point
    if s_end >= 1.0:
        result.append(polyline[-1].copy())
    else:
        seg_idx = np.searchsorted(cum_norm, s_end) - 1
        seg_idx = max(0, min(len(polyline) - 2, seg_idx))
        t = (s_end - cum_norm[seg_idx]) / (cum_norm[seg_idx + 1] - cum_norm[seg_idx] + 1e-9)
        pt = polyline[seg_idx] + t * (polyline[seg_idx + 1] - polyline[seg_idx])
        result.append(pt)

    return np.array(result)


# ---------------------------------------------------------------------------
# Full map parsing pipeline
# ---------------------------------------------------------------------------

def parse_map(map_path: str, target_hw: Optional[tuple] = None,
              route_polyline: Optional[np.ndarray] = None,
              route_progress: Optional[tuple] = None) -> dict:
    """Full map parsing: semantic channels + road skeleton + junctions + route.

    Args:
        map_path: path to map image.
        target_hw: (H, W) to resize map to. None = original size.
        route_polyline: optional [N, 2] route in map pixel coords.
        route_progress: optional (start, end) progress [0,1] for visible segment.

    Returns:
        dict with:
            'semantic': [H, W, 4] (water/road/building/green)
            'road_mask': [H, W] uint8
            'skeleton': [H, W] uint8
            'junction_mask': [H, W] uint8
            'junction_points': list of (x, y)
            'route_mask': [H, W] uint8 (full route, if provided)
            'route_visible_mask': [H, W] uint8 (visible segment, if progress given)
            'topology_channels': [H, W, 3] (road, junction, route_visible)
    """
    sem = extract_semantic_channels(map_path, target_hw)
    H, W = sem.shape[:2]

    road_mask = extract_road_mask(map_path, (H, W))
    skeleton = skeletonize_road(road_mask)
    junction_mask, junction_points = detect_junctions(road_mask)

    result = {
        'semantic': sem,
        'road_mask': road_mask,
        'skeleton': skeleton,
        'junction_mask': junction_mask,
        'junction_points': junction_points,
    }

    if route_polyline is not None:
        # Scale route to target size if needed
        route = route_polyline.copy().astype(np.float32)
        route_full_mask = rasterize_route(route, (H, W))
        result['route_mask'] = route_full_mask

        if route_progress is not None:
            vis_poly = route_progress_segment(route, route_progress[0], route_progress[1])
            route_vis_mask = rasterize_route(vis_poly, (H, W), thickness=5)
            result['route_visible_mask'] = route_vis_mask
        else:
            result['route_visible_mask'] = route_full_mask.copy()

    # Combined topology channels: road, junction, route_visible
    topo = np.stack([
        (road_mask > 127).astype(np.float32),
        (junction_mask > 127).astype(np.float32),
        (result.get('route_visible_mask', np.zeros((H, W), dtype=np.uint8)) > 127).astype(np.float32),
    ], axis=-1)
    result['topology_channels'] = topo

    return result
