import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis

from .feature_extractor import extract_features


def bbox_to_kalman(bbox):
    """
    input as x_min,y_min,x_max,y_max
    output as x_centre,y_centre,size,ascpect ratio
    """
    width, height = bbox[2:4] - bbox[0:2]
    center_x, center_y = (bbox[0:2] + bbox[2:4]) / 2
    area = width * height  # scale is just area
    r = width / height
    out = np.array([center_x, center_y, area, r]).astype(np.float64)
    return np.expand_dims(out, axis=1)


def kalman_to_bbox(bbox):
    """
    input as x_centre,y_centre,size,ascpect ratio
    output as x_min,y_min,x_max,y_max
    """
    bbox = bbox[:, 0]
    width = np.sqrt(bbox[2] * bbox[3])
    height = bbox[2] / width
    x_min, y_min, x_max, y_max = (
        bbox[0] - width / 2,
        bbox[1] - height / 2,
        bbox[0] + width / 2,
        bbox[1] + height / 2,
    )

    return np.array([x_min, y_min, x_max, y_max]).astype(np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    calculate iou between two boxes
    """
    a_tl, a_br = a[:4].reshape((2, 2))
    b_tl, b_br = b[:4].reshape((2, 2))
    int_tl = np.maximum(a_tl, b_tl)
    int_br = np.minimum(a_br, b_br)
    int_area = np.product(np.maximum(0.0, int_br - int_tl))
    a_area = np.product(a_br - a_tl)
    b_area = np.product(b_br - b_tl)
    return int_area / (a_area + b_area - int_area)


def compute_mahalanobis(a: np.ndarray, b: np.ndarray):
    # matrix = np.concatenate((a, b), axis=0)
    a = a
    b = b
    cov = np.cov(b)
    # if np.sum(np.cov(cov)) == 0:
    #     return 0

    dist = np.sqrt(np.matmul((a - b) * cov, (a - b).T))
    return dist


def create_deep_matrix(detections, trackers, det_features, trk_features, coeff):
    dist_matrix = np.zeros(shape=(len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_value = iou(det, trk)
            dist_value = compute_mahalanobis(det_features[d], trk_features[t])
            matrix_elem = coeff * dist_value - (1 - coeff) * iou_value
            dist_matrix[d, t] = matrix_elem

    return dist_matrix


def compare_boxes_deep(detections, trackers, det_features, trk_features, thresh, coeff=0.5):
    dist_matrix = create_deep_matrix(detections=detections,
                                     trackers=trackers,
                                     det_features=det_features,
                                     trk_features=trk_features,
                                     coeff=coeff,
                                     )
    return find_indices(dist_matrix, thresh, len(detections), len(trackers))


def find_indices(dist_matrix, thresh, det_len, trk_len):
    row_id, col_id = linear_sum_assignment(-dist_matrix)
    matched_indices = np.transpose(np.array([row_id, col_id]))

    matrix_values = np.array(
        [dist_matrix[row_id, col_id] for row_id, col_id in matched_indices]
    )
    best_indices = matched_indices[matrix_values > thresh]

    unmatched_detection_indices = np.array(
        [d for d in range(det_len) if d not in best_indices[:, 0]]
    )
    unmatched_trackers_indices = np.array(
        [t for t in range(trk_len) if t not in best_indices[:, 1]]
    )

    return best_indices, unmatched_detection_indices, unmatched_trackers_indices


def compare_boxes(detections, trackers, iou_thresh=0.3):
    iou_matrix = np.zeros(shape=(len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    return find_indices(iou_matrix, iou_thresh, len(detections), len(trackers))
