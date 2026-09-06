"""
change_detection.py
--------------------
Portable version of the notebook's Section 8-10b pipeline (camera-shift compensation,
Hungarian + appearance-gated matching, soft net-count heuristic, RandomForest classifier),
so the Flask app produces IDENTICAL results to the notebook -- not a re-implementation
that could quietly drift out of sync.

Everything here is stateless / pure functions except `ChangeDetector`, which loads the
YOLO weights + RandomForest model + config once and reuses them across requests.
"""
import os
import json
import tempfile
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from scipy.optimize import linear_sum_assignment
from ultralytics import RTDETR, YOLO


# ---------------------------------------------------------------------------
# Preprocessing -- MUST match Section 3a of the notebook exactly, since the
# YOLO model was trained on CLAHE-equalized images. Skipping this at inference
# time would feed the model images from a different distribution than training.
# ---------------------------------------------------------------------------
def equalize_clahe(img_bgr, clip_limit=2.5, tile_grid_size=(8, 8)):
    """Contrast-Limited Adaptive Histogram Equalization on the L channel only
    (LAB color space) -- boosts local contrast from uneven room lighting without
    distorting color balance."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def prepare_image_for_pipeline(path, tmp_dir=None, clip_limit=2.5, tile_grid_size=(8, 8)):
    """Loads an arbitrary (freshly uploaded, never-before-seen) image, applies the same
    CLAHE preprocessing the model was trained on, and writes it to a temp file. Returns
    that temp path so the rest of the pipeline (detection, homography, appearance crops)
    keeps working on plain file paths exactly as before."""
    tmp_dir = tmp_dir or tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    eq = equalize_clahe(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    out_path = os.path.join(tmp_dir, f"eq_{os.path.basename(path)}")
    cv2.imwrite(out_path, eq)
    return out_path


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def extract_boxes(results, model):
    boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = model.names.get(cls_id)
            if class_name is None:
                warnings.warn(
                    f'Skipping detection with unknown YOLO class id {cls_id}; '
                    f'model defines {len(model.names)} classes.',
                    RuntimeWarning,
                )
                continue
            boxes.append({
                'class': class_name,
                'bbox': xyxy,
                'confidence': float(box.conf[0].cpu().numpy())
            })
    return boxes


def get_image_size(path):
    with PILImage.open(path) as im:
        return im.size


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Camera-shift compensation (ORB + RANSAC homography)
# ---------------------------------------------------------------------------
def estimate_camera_shift(before_path, after_path, max_features=2000, good_match_pct=0.2,
                           ransac_reproj_thresh=5.0):
    img1 = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return np.eye(3), 0.0

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return np.eye(3), 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    n_good = max(15, int(len(matches) * good_match_pct))
    matches = matches[:n_good]
    if len(matches) < 8:
        return np.eye(3), 0.0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=ransac_reproj_thresh)
    if H is None:
        return np.eye(3), 0.0

    inliers = inlier_mask.ravel().astype(bool)
    if inliers.sum() < 8:
        return np.eye(3), 0.0

    pts1_proj = cv2.perspectiveTransform(pts1[inliers], H)
    residuals = np.linalg.norm((pts1_proj - pts2[inliers]).reshape(-1, 2), axis=1)
    return H, float(np.median(residuals))


def adaptive_move_thresh(residual_px, min_thresh=15.0, safety_factor=3.0):
    return max(min_thresh, safety_factor * residual_px)


def warp_bbox(bbox, H):
    if H is None:
        return bbox
    x1, y1, x2, y2 = bbox
    corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    wx1, wy1 = float(warped[:, 0].min()), float(warped[:, 1].min())
    wx2, wy2 = float(warped[:, 0].max()), float(warped[:, 1].max())
    return [wx1, wy1, wx2, wy2]


# ---------------------------------------------------------------------------
# Appearance similarity (disambiguates real moves from unrelated same-class objects)
# ---------------------------------------------------------------------------
def crop_bbox(img, bbox, pad=4):
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def appearance_similarity(crop_a, crop_b):
    if crop_a is None or crop_b is None or crop_a.size == 0 or crop_b.size == 0:
        return 0.0
    hsv_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(crop_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return float(max(0.0, corr))


# ---------------------------------------------------------------------------
# Matching + classification -- identical logic to the notebook (Section 8/8a/8b)
# ---------------------------------------------------------------------------
def match_boxes(before_boxes, after_boxes, H=None, move_thresh=50,
                 image_diag=None, max_match_dist_frac=0.5,
                 before_img=None, after_img=None, min_appearance_sim=0.35):
    events = []
    matched_pairs = []
    max_match_dist = (max_match_dist_frac * image_diag) if image_diag else float('inf')
    use_appearance = before_img is not None and after_img is not None

    classes = set(b['class'] for b in before_boxes) | set(a['class'] for a in after_boxes)

    for cls in classes:
        b_idx = [i for i, b in enumerate(before_boxes) if b['class'] == cls]
        a_idx = [j for j, a in enumerate(after_boxes) if a['class'] == cls]
        if not b_idx and not a_idx:
            continue

        b_centers = []
        for i in b_idx:
            wb = warp_bbox(before_boxes[i]['bbox'], H)
            b_centers.append(((wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2))
        a_centers = []
        for j in a_idx:
            ab = after_boxes[j]['bbox']
            a_centers.append(((ab[0] + ab[2]) / 2, (ab[1] + ab[3]) / 2))

        matched_b, matched_a = set(), set()

        if b_idx and a_idx:
            cost = np.zeros((len(b_idx), len(a_idx)))
            for bi, (bx, by) in enumerate(b_centers):
                for aj, (ax, ay) in enumerate(a_centers):
                    cost[bi, aj] = np.hypot(ax - bx, ay - by)
            row_ind, col_ind = linear_sum_assignment(cost)

            for bi, aj in zip(row_ind, col_ind):
                dist = float(cost[bi, aj])
                if dist > max_match_dist:
                    continue

                i, j = b_idx[bi], a_idx[aj]
                b, a = before_boxes[i], after_boxes[j]

                sim = 1.0
                if use_appearance:
                    crop_b_img = crop_bbox(before_img, b['bbox'])
                    crop_a_img = crop_bbox(after_img, a['bbox'])
                    sim = appearance_similarity(crop_b_img, crop_a_img)
                    if sim < min_appearance_sim:
                        continue

                matched_b.add(bi)
                matched_a.add(aj)
                matched_pairs.append((b, a, dist, sim))
                if dist > move_thresh:
                    events.append({'type': 'Move', 'object': cls,
                                    'confidence': (b['confidence'] + a['confidence']) / 2,
                                    'movement': dist, 'appearance_sim': sim})

        for bi, i in enumerate(b_idx):
            if bi not in matched_b:
                events.append({'type': 'Delete', 'object': cls,
                                'confidence': before_boxes[i]['confidence'], 'movement': 0.0})
        for aj, j in enumerate(a_idx):
            if aj not in matched_a:
                events.append({'type': 'Add', 'object': cls,
                                'confidence': after_boxes[j]['confidence'], 'movement': 0.0})

    return events, matched_pairs


def summarize_events(events, matched_pairs=None):
    matched_pairs = matched_pairs or []
    add_events = [e for e in events if e['type'] == 'Add']
    del_events = [e for e in events if e['type'] == 'Delete']
    move_events = [e for e in events if e['type'] == 'Move']

    def best(evs):
        return max(evs, key=lambda e: e['confidence']) if evs else None

    best_add, best_del, best_move = best(add_events), best(del_events), best(move_events)
    max_matched_dist = max((d for _, _, d, _ in matched_pairs), default=0.0)
    avg_appearance_sim = float(np.mean([s for _, _, _, s in matched_pairs])) if matched_pairs else 0.0

    return {
        'n_add': len(add_events), 'n_del': len(del_events), 'n_move': len(move_events),
        'net': len(add_events) - len(del_events),
        'best_add_conf': best_add['confidence'] if best_add else 0.0,
        'best_del_conf': best_del['confidence'] if best_del else 0.0,
        'best_move_conf': best_move['confidence'] if best_move else 0.0,
        'best_move_dist': best_move['movement'] if best_move else 0.0,
        'max_matched_dist': max_matched_dist,
        'avg_appearance_sim': avg_appearance_sim,
        'best_add': best_add, 'best_del': best_del, 'best_move': best_move,
    }


def pair_level_label(events, matched_pairs=None):
    s = summarize_events(events, matched_pairs)
    candidates = [c for c in [s['best_add'], s['best_del'], s['best_move']] if c is not None]

    if abs(s['net']) <= 1 and s['best_move'] is not None:
        winner = max(candidates, key=lambda e: e['confidence'])
        return winner['type'], winner['object']

    if s['net'] > 0 and s['best_add']:
        return 'Add', s['best_add']['object']
    if s['net'] < 0 and s['best_del']:
        return 'Delete', s['best_del']['object']

    if s['best_move']:
        return 'Move', s['best_move']['object']
    if matched_pairs:
        b, a, dist, sim = max(matched_pairs, key=lambda t: t[2])
        return 'Move', b['class']

    if candidates:
        winner = max(candidates, key=lambda e: e['confidence'])
        return winner['type'], winner['object']
    return 'Move', 'Unknown'


def object_for_type(predicted_type, stats, matched_pairs):
    key = {'Add': 'best_add', 'Delete': 'best_del', 'Move': 'best_move'}.get(predicted_type)
    event = stats.get(key) if key else None
    if event:
        return event['object']
    if matched_pairs:
        before_box, _, _, _ = max(matched_pairs, key=lambda pair: pair[2])
        return before_box['class']
    return 'Unknown'


def build_feature_dict(before_boxes, after_boxes, ious, residual_px, move_thresh,
                       events, matched_pairs):
    stats = summarize_events(events, matched_pairs)
    best_move = stats['best_move']
    best_move_appearance_sim = best_move['appearance_sim'] if best_move else 0.0
    n_matched = len(matched_pairs)
    frac_matched_before = n_matched / len(before_boxes) if before_boxes else 0.0
    frac_matched_after = n_matched / len(after_boxes) if after_boxes else 0.0
    n_classes_changed = len({event['object'] for event in events})

    area_ratios = []
    for before_box, after_box, _, _ in matched_pairs:
        before_bbox, after_bbox = before_box['bbox'], after_box['bbox']
        before_area = max(1e-6, (before_bbox[2] - before_bbox[0]) *
                          (before_bbox[3] - before_bbox[1]))
        after_area = max(1e-6, (after_bbox[2] - after_bbox[0]) *
                           (after_bbox[3] - after_bbox[1]))
        area_ratios.append(min(before_area, after_area) / max(before_area, after_area))

    total_events = stats['n_add'] + stats['n_del'] + stats['n_move']
    net_count_diff = len(after_boxes) - len(before_boxes)
    features = {
        'before_count': len(before_boxes), 'after_count': len(after_boxes),
        'net_count_diff': net_count_diff,
        'total_objects': len(before_boxes) + len(after_boxes),
        'max_iou': max(ious) if ious else 0.0,
        'avg_iou': float(np.mean(ious)) if ious else 0.0,
        'camera_shift_residual_px': residual_px, 'move_thresh_used': move_thresh,
        'n_add': stats['n_add'], 'n_del': stats['n_del'], 'n_move': stats['n_move'],
        'best_add_conf': stats['best_add_conf'], 'best_del_conf': stats['best_del_conf'],
        'best_move_conf': stats['best_move_conf'], 'best_move_dist': stats['best_move_dist'],
        'max_matched_dist': stats['max_matched_dist'],
        'avg_appearance_sim': stats['avg_appearance_sim'],
        'best_move_appearance_sim': best_move_appearance_sim,
        'n_matched': n_matched, 'frac_matched_before': frac_matched_before,
        'frac_matched_after': frac_matched_after,
        'add_del_conf_margin': stats['best_add_conf'] - stats['best_del_conf'],
        'move_conf_margin': stats['best_move_conf'] - max(stats['best_add_conf'], stats['best_del_conf']),
        'n_classes_changed': n_classes_changed,
        'avg_matched_area_ratio': float(np.mean(area_ratios)) if area_ratios else 1.0,
        'total_events': total_events,
        'move_event_ratio': stats['n_move'] / total_events if total_events else 0.0,
        'is_net_ambiguous': 1.0 if abs(net_count_diff) <= 1 else 0.0,
    }
    return features, stats


# ---------------------------------------------------------------------------
# Visualization helper for the web UI
# ---------------------------------------------------------------------------
def draw_boxes(img_bgr, boxes, color=(0, 200, 0)):
    out = img_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b['bbox']]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{b['class']} {b['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


# ---------------------------------------------------------------------------
# ChangeDetector: loads everything once, exposes a single analyze() call
# ---------------------------------------------------------------------------
class ChangeDetector:
    def __init__(self, model_dir):
        """model_dir must contain: best.pt, rf_classifier.joblib, config.json
        (all produced by the notebook's 'Export for Flask Deployment' section)."""
        with open(os.path.join(model_dir, 'config.json')) as f:
            self.config = json.load(f)

        weights_path = os.path.join(model_dir, 'best.pt')
        try:
            self.yolo = RTDETR(weights_path)
        except Exception:
            self.yolo = YOLO(weights_path)

        rf_path = os.path.join(model_dir, 'rf_classifier.joblib')
        self.rf = None
        if os.path.exists(rf_path):
            try:
                self.rf = joblib.load(rf_path)
            except Exception as exc:
                warnings.warn(
                    f'Could not load RandomForest classifier; using heuristic fallback: {exc}',
                    RuntimeWarning,
                )

    def analyze(self, before_path, after_path, tmp_dir=None):
        cfg = self.config
        before_eq = prepare_image_for_pipeline(before_path, tmp_dir,
                                                 cfg.get('clahe_clip_limit', 2.5),
                                                 tuple(cfg.get('clahe_tile_grid', [8, 8])))
        after_eq = prepare_image_for_pipeline(after_path, tmp_dir,
                                               cfg.get('clahe_clip_limit', 2.5),
                                               tuple(cfg.get('clahe_tile_grid', [8, 8])))

        conf_thresh = cfg.get('conf_thresh', 0.4)
        before_res = self.yolo(before_eq, conf=conf_thresh, verbose=False)
        after_res = self.yolo(after_eq, conf=conf_thresh, verbose=False)
        before_boxes = extract_boxes(before_res, self.yolo)
        after_boxes = extract_boxes(after_res, self.yolo)

        H, residual_px = estimate_camera_shift(before_eq, after_eq)
        move_thresh = adaptive_move_thresh(
            residual_px, cfg.get('move_min_thresh', 15.0), cfg.get('move_safety_factor', 3.0))

        after_w, after_h = get_image_size(after_eq)
        image_diag = float(np.hypot(after_w, after_h))

        before_img = cv2.imread(before_eq)
        after_img = cv2.imread(after_eq)

        events, matched_pairs = match_boxes(
            before_boxes, after_boxes, H=H, move_thresh=move_thresh, image_diag=image_diag,
            max_match_dist_frac=cfg.get('max_match_dist_frac', 0.5),
            before_img=before_img, after_img=after_img,
            min_appearance_sim=cfg.get('min_appearance_sim', 0.35))
        ious = [compute_iou(warp_bbox(before_box['bbox'], H), after_box['bbox'])
                for before_box, after_box, _, _ in matched_pairs]
        features, stats = build_feature_dict(
            before_boxes, after_boxes, ious, residual_px, move_thresh, events, matched_pairs)

        predicted, predicted_object, used_model = None, None, 'heuristic'
        if self.rf is not None:
            feature_cols = cfg.get('feature_cols', list(features))
            x_row = pd.DataFrame([[features[c] for c in feature_cols]], columns=feature_cols)
            predicted = self.rf.predict(x_row)[0]
            predicted_object = object_for_type(predicted, stats, matched_pairs)
            used_model = 'random_forest'
        else:
            predicted, predicted_object = pair_level_label(events, matched_pairs)

        return {
            'predicted_change': str(predicted),
            'predicted_object': str(predicted_object),
            'model_used': used_model,
            'before_boxes': before_boxes,
            'after_boxes': after_boxes,
            'camera_shift_residual_px': residual_px,
            'move_thresh_used': move_thresh,
            'stats': {k: v for k, v in stats.items() if k not in ('best_add', 'best_del', 'best_move')},
            'before_eq_path': before_eq,
            'after_eq_path': after_eq,
        }
