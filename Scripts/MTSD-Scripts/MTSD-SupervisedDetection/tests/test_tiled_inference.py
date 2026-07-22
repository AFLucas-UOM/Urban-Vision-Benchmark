from run_tiled_inference import merge_class_aware_nms, tile_origins, tile_windows


def test_tile_origins_cover_edges_with_overlap():
    assert tile_origins(1000, 1280, 0.2) == [0]
    origins = tile_origins(3000, 1280, 0.2)
    assert origins[0] == 0 and origins[-1] == 1720
    assert all(right > left for left, right in zip(origins, origins[1:]))
    windows = tile_windows(3000, 2000, 1280, 0.2)
    assert windows[0] == (0, 0, 1280, 1280)
    assert windows[-1] == (1720, 720, 3000, 2000)


def test_class_aware_nms_keeps_overlapping_boxes_from_different_classes():
    candidates = [
        {"xyxy": [0, 0, 10, 10], "score": .9, "category_id": 0},
        {"xyxy": [1, 1, 11, 11], "score": .8, "category_id": 0},
        {"xyxy": [1, 1, 11, 11], "score": .7, "category_id": 1},
    ]
    kept = merge_class_aware_nms(candidates, iou_threshold=.5, max_detections=100)
    assert [(row["category_id"], row["score"]) for row in kept] == [(0, .9), (1, .7)]
