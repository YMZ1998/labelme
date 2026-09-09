from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import draw
from skimage import filters
from skimage import measure


def fit_imaging_circle(gray: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Fit the imaging field's outer circle, excluding text and clipped borders."""
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    level = background + max(
        1.0, (float(filters.threshold_otsu(gray)) - background) * 0.35
    )
    component = _largest_component(gray > level)
    solid = ndimage.binary_fill_holes(component)
    contour = max(mask_contours(solid), key=len)
    height, width = gray.shape
    margin = 2
    points = contour[
        (contour[:, 0] > margin)
        & (contour[:, 0] < width - margin)
        & (contour[:, 1] > margin)
        & (contour[:, 1] < height - margin)
    ]
    minimum_points = 30
    if len(points) < minimum_points:
        raise ValueError("No circular imaging boundary found")
    for _ in range(5):
        design = np.column_stack((2 * points, np.ones(len(points))))
        parameters, _, rank, _ = np.linalg.lstsq(
            design, (points**2).sum(axis=1), rcond=None
        )
        if rank < design.shape[1]:
            raise ValueError("Cannot fit imaging circle")
        center = parameters[:2]
        radius = math.sqrt(max(0, float(parameters[2] + center @ center)))
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        points = points[residual <= max(1.0, float(np.quantile(residual, 0.85)))]
    if not (
        0 < center[0] < width and 0 < center[1] < height and radius > minimum_points / 3
    ):
        raise ValueError("Invalid imaging circle")
    return _refine_outer_radius(
        gray, circle=(float(center[0]), float(center[1]), radius)
    )


def _refine_outer_radius(
    gray: npt.NDArray[np.float64], *, circle: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Move a threshold-based radius to the outer radial intensity edge."""
    center_x, center_y, radius = circle
    angles = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    radii = np.linspace(radius * 0.88, radius * 1.18, 180)
    sample_x = center_x + np.cos(angles[:, None]) * radii
    sample_y = center_y + np.sin(angles[:, None]) * radii
    height, width = gray.shape
    valid = (
        (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
    )
    samples = ndimage.map_coordinates(
        gray,
        [sample_y, sample_x],
        order=1,
        mode="nearest",
    )
    samples[~valid] = np.nan
    with np.errstate(all="ignore"):
        profile = np.nanmedian(samples, axis=0)
    finite = np.isfinite(profile)
    if np.count_nonzero(finite) < len(profile) * 0.8:
        return circle
    profile = np.interp(radii, radii[finite], profile[finite])
    gradient = np.gradient(ndimage.gaussian_filter1d(profile, sigma=2.5))
    edge_index = int(np.argmin(gradient))
    edge_radius = float(radii[edge_index])
    contrast = float(np.nanpercentile(samples, 75) - np.nanpercentile(samples, 25))
    if -gradient[edge_index] < max(0.25, contrast * 0.015):
        return circle
    return center_x, center_y, edge_radius


def estimate_inner_radius(
    gray: npt.NDArray[np.float64], circle: tuple[float, float, float]
) -> float:
    polar, radii = _polar_image(gray, circle)
    profile = np.median(polar, axis=0)
    gradient = np.gradient(ndimage.gaussian_filter1d(profile, 2))
    valid = (radii > circle[2] * 0.08) & (radii < circle[2] * 0.55)
    return float(radii[np.argmax(np.where(valid, gradient, -np.inf))])


def _polar_image(
    gray: npt.NDArray[np.float64], circle: tuple[float, float, float]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    x, y, radius = circle
    angles = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    radii = np.arange(1, max(3, int(radius * 0.75)), dtype=float)
    polar = ndimage.map_coordinates(
        gray,
        [y + np.sin(angles[:, None]) * radii, x + np.cos(angles[:, None]) * radii],
        order=1,
        mode="nearest",
    )
    return polar, radii


def trace_imaging_ring(
    gray: npt.NDArray[np.float64],
    *,
    circle: tuple[float, float, float],
    inner_radius: float,
    smoothness: float = 0.5,
) -> npt.NDArray[np.bool_]:
    """Trace a continuous inner edge in polar space; use a fitted outer circle."""
    polar, radii = _polar_image(gray, circle)
    gradient = np.gradient(polar, axis=1)
    band = np.abs(radii - inner_radius) <= max(8, circle[2] * 0.12)
    radii = radii[band]
    gradient = gradient[:, band]
    if radii.size <= 1:
        raise ValueError("Inner search band is empty")
    strength = np.maximum(gradient, 0)
    normalizer = max(float(np.quantile(strength, 0.95)), 0.1)
    costs = (
        -strength / normalizer
        + 0.08 * ((radii - inner_radius) / max(circle[2] * 0.1, 1)) ** 2
    )
    # Start in a high-contrast direction and enforce a closed path there.
    start_angle = int(np.argmax(strength.max(axis=1)))
    costs = np.roll(costs, -start_angle, axis=0)
    seed = int(np.argmin(costs[0]))
    scores = np.full(len(radii), np.inf)
    scores[seed] = costs[0, seed]
    back = np.zeros(costs.shape, dtype=np.int32)
    max_step = max(2, int(circle[2] / 100))
    indices = np.arange(len(radii))
    steps = np.arange(-max_step, max_step + 1)
    previous = indices[None, :] + steps[:, None]
    valid = (previous >= 0) & (previous < len(radii))
    previous = np.clip(previous, 0, len(radii) - 1)
    for angle in range(1, len(costs)):
        candidates = scores[previous] + smoothness * steps[:, None] ** 2
        candidates[~valid] = np.inf
        choice = candidates.argmin(axis=0)
        back[angle] = previous[choice, indices]
        scores = candidates[choice, indices] + costs[angle]
    path = np.empty(len(costs), dtype=np.int32)
    path[-1] = seed
    for angle in range(len(path) - 1, 0, -1):
        path[angle - 1] = back[angle, path[angle]]
    inner = np.roll(radii[path], start_angle)
    inner = ndimage.gaussian_filter1d(inner, 1, mode="wrap")
    yy, xx = np.indices(gray.shape)
    distance = np.hypot(xx - circle[0], yy - circle[1])
    angles = np.arctan2(yy - circle[1], xx - circle[0]) % (2 * math.pi)
    local_inner = np.interp(
        angles, np.linspace(0, 2 * math.pi, len(inner) + 1), np.r_[inner, inner[0]]
    )
    return (distance <= circle[2]) & (distance >= local_inner)


def ring_grayscale(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    grayscale_dimensions = 2
    gray = image.astype(np.float64)
    if gray.ndim > grayscale_dimensions:
        gray = gray[..., :3] @ np.array([0.299, 0.587, 0.114])
    return ndimage.gaussian_filter(gray, sigma=1.5)


def trace_default_imaging_ring(
    image: npt.NDArray[np.uint8],
) -> npt.NDArray[np.bool_]:
    """Extract a ring with automatically estimated inner and outer boundaries."""
    gray = ring_grayscale(image)
    circle = fit_imaging_circle(gray)
    inner_radius = estimate_inner_radius(gray, circle)
    return trace_imaging_ring(
        gray,
        circle=circle,
        inner_radius=inner_radius,
        smoothness=0.5,
    )


def _largest_component(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("No foreground")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def mask_contours(mask: npt.NDArray[np.bool_]) -> list[npt.NDArray[np.float64]]:
    contours = measure.find_contours(np.pad(mask, 1), 0.5)
    return [contour[:, ::-1] - 1 for contour in contours]


def resample_closed_contour(
    contour: npt.ArrayLike, *, point_spacing: float
) -> npt.NDArray[np.float64]:
    """Resample a closed contour at approximately equal pixel intervals."""
    coordinate_count = 2
    minimum_polygon_points = 3
    points = np.asarray(contour, dtype=np.float64)
    if (
        points.ndim != coordinate_count
        or points.shape[1] != coordinate_count
        or len(points) < minimum_polygon_points
    ):
        raise ValueError("Expected at least three contour points")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(point_spacing)
        or point_spacing <= 0
    ):
        raise ValueError("Point spacing must be positive and finite")
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    closed = np.vstack((points, points[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    perimeter = float(cumulative[-1])
    if perimeter == 0:
        raise ValueError("Contour has zero perimeter")
    point_count = max(minimum_polygon_points, math.ceil(perimeter / point_spacing))
    distances = np.linspace(0, perimeter, point_count, endpoint=False)
    segment_indices = np.searchsorted(cumulative, distances, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(points) - 1)
    local = distances - cumulative[segment_indices]
    lengths = segment_lengths[segment_indices]
    fractions = np.divide(local, lengths, out=np.zeros_like(local), where=lengths > 0)
    return closed[segment_indices] + fractions[:, None] * (
        closed[segment_indices + 1] - closed[segment_indices]
    )


def _resample_open_contour(
    contour: npt.NDArray[np.float64], *, point_spacing: float
) -> npt.NDArray[np.float64]:
    """Resample an open contour while preserving both endpoints."""
    segment_lengths = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    length = float(cumulative[-1])
    if length == 0:
        return contour[:1]
    segment_count = max(1, math.ceil(length / point_spacing))
    distances = np.linspace(0, length, segment_count + 1)
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(contour) - 2)
    local = distances - cumulative[indices]
    lengths = segment_lengths[indices]
    fractions = np.divide(local, lengths, out=np.zeros_like(local), where=lengths > 0)
    sampled = contour[indices] + fractions[:, None] * (
        contour[indices + 1] - contour[indices]
    )
    sampled[-1] = contour[-1]
    return sampled


def _resample_cut_contour(
    contour: npt.NDArray[np.float64],
    controls: npt.NDArray[np.float64],
    *,
    point_spacing: float,
) -> npt.NDArray[np.float64]:
    """Sample both curved boundaries by spacing and each cutting side at 3 points."""
    if np.array_equal(contour[0], contour[-1]):
        contour = contour[:-1]
    anchors = np.array(
        [np.linalg.norm(contour - point, axis=1).argmin() for point in controls]
    )
    if len(np.unique(anchors)) != len(controls):
        raise ValueError("Cutting points are too close together")

    # The four anchors divide the closed outline into two arcs and two radial sides.
    # Labels 0-1 and 2-3 are the sides selected by the first and second point pairs.
    label_at_anchor = {int(index): label for label, index in enumerate(anchors)}
    ordered = sorted(label_at_anchor)
    side_pairs = {frozenset((0, 1)), frozenset((2, 3))}
    sampled_parts: list[npt.NDArray[np.float64]] = []
    for position, start_index in enumerate(ordered):
        end_index = ordered[(position + 1) % len(ordered)]
        if end_index > start_index:
            part = contour[start_index : end_index + 1]
        else:
            part = np.vstack((contour[start_index:], contour[: end_index + 1]))
        labels = frozenset(
            (label_at_anchor[start_index], label_at_anchor[end_index])
        )
        if labels in side_pairs:
            sampled = np.linspace(part[0], part[-1], 3)
        else:
            sampled = _resample_open_contour(part, point_spacing=point_spacing)
        sampled_parts.append(sampled[:-1])
    return np.vstack(sampled_parts)


def cut_ring(
    mask: npt.NDArray[np.bool_],
    points: npt.ArrayLike,
    *,
    major_arc: bool = True,
    point_spacing: float = 12.0,
) -> npt.NDArray[np.float64]:
    """Cut the segmented mask using the two radial sides selected by four clicks."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Expected four points")
    outer_start, inner_start, inner_end, outer_end = points
    first, second = outer_start - inner_start, outer_end - inner_end
    widths = np.linalg.norm([first, second], axis=1)
    epsilon = 1e-6
    if np.any(widths < epsilon):
        raise ValueError("Zero-width side")
    axes = np.column_stack((first / widths[0], -second / widths[1]))
    minimum_sine = 1e-3
    if abs(np.linalg.det(axes)) < minimum_sine:
        region = _cut_ring_between_parallel_sides(
            mask=mask,
            controls=points,
            major_arc=major_arc,
        )
        return _ring_region_to_polygon(
            region=region,
            controls=points,
            point_spacing=point_spacing,
        )
    distances = np.linalg.solve(axes, inner_end - inner_start)
    if np.any(distances >= -epsilon):
        raise ValueError("Reversed inner/outer points")
    center = inner_start + distances[0] * axes[:, 0]
    hole = ndimage.binary_fill_holes(mask) & ~mask
    x, y = np.rint(center).astype(int)
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and hole[y, x]):
        raise ValueError("The radial sides must meet inside the ring hole")
    start = math.atan2(*(outer_start - center)[::-1])
    end = math.atan2(*(outer_end - center)[::-1])
    sweep = (end - start + math.pi) % (2 * math.pi) - math.pi
    if major_arc:
        sweep -= math.copysign(2 * math.pi, sweep)
    yy, xx = np.indices(mask.shape)
    angle = np.arctan2(yy - center[1], xx - center[0])
    progress = ((angle - start) * np.sign(sweep)) % (2 * math.pi)
    region = mask & (progress <= abs(sweep))
    if not region.any():
        raise ValueError("Empty cut")
    labels, count = ndimage.label(region)
    if count != 1:
        raise ValueError("Cut is disconnected; adjust threshold or radial sides")
    return _ring_region_to_polygon(
        region=labels != 0,
        controls=points,
        point_spacing=point_spacing,
    )


def _cut_ring_between_parallel_sides(
    *,
    mask: npt.NDArray[np.bool_],
    controls: npt.NDArray[np.float64],
    major_arc: bool,
) -> npt.NDArray[np.bool_]:
    """Split a ring along arbitrary parallel sides and select one component."""
    separator = np.zeros(mask.shape, dtype=np.bool_)
    extension = 3.0
    for outer, inner in ((controls[0], controls[1]), (controls[3], controls[2])):
        direction = outer - inner
        direction /= np.linalg.norm(direction)
        start = np.rint(inner - direction * extension).astype(int)
        end = np.rint(outer + direction * extension).astype(int)
        rows, columns = draw.line(start[1], start[0], end[1], end[0])
        valid = (
            (rows >= 0)
            & (rows < mask.shape[0])
            & (columns >= 0)
            & (columns < mask.shape[1])
        )
        separator[rows[valid], columns[valid]] = True
    separator = ndimage.binary_dilation(separator, iterations=2)

    labels, count = ndimage.label(mask & ~separator)
    minimum_components = 2
    if count < minimum_components:
        raise ValueError("The cutting sides do not split the ring")
    sizes = np.bincount(labels.ravel())
    component_labels = np.argsort(sizes[1:])[::-1] + 1
    selected_label = component_labels[0 if major_arc else 1]
    core = labels == selected_label
    restored_side = separator & ndimage.binary_dilation(core, iterations=2)
    return core | restored_side


def _ring_region_to_polygon(
    *,
    region: npt.NDArray[np.bool_],
    controls: npt.NDArray[np.float64],
    point_spacing: float,
) -> npt.NDArray[np.float64]:
    contours = mask_contours(region)
    if len(contours) != 1:
        raise ValueError("Cut must open the ring hole")
    if not np.isfinite(point_spacing) or point_spacing <= 0:
        raise ValueError("Point spacing must be positive and finite")
    polygon = _resample_cut_contour(
        contours[0], controls, point_spacing=point_spacing
    )
    polygon[:, 0] = np.clip(polygon[:, 0], 0, region.shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, region.shape[0] - 1)
    return polygon
