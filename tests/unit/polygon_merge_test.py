import pytest

from labelme._polygon_merge import merge_polygons


def test_merge_overlapping_polygons_returns_one_outline() -> None:
    merged = merge_polygons(
        [[2, 2], [20, 2], [20, 20], [2, 20]],
        [[12, 12], [30, 12], [30, 30], [12, 30]],
        image_shape=(40, 40),
        detail=100,
    )

    assert merged.shape[1] == 2
    assert len(merged) >= 4
    assert merged[:, 0].min() <= 2
    assert merged[:, 0].max() >= 30


def test_merge_disconnected_polygons_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap or touch"):
        merge_polygons(
            [[2, 2], [8, 2], [8, 8], [2, 8]],
            [[20, 20], [28, 20], [28, 28], [20, 28]],
            image_shape=(32, 32),
        )
