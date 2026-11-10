#!/usr/bin/env python3
"""Uji akurasi deteksi HSV (make_mask + find_hand_centers) dari color_control.py.

Tanpa kamera fisik: frame sintetis dirender (latar, objek non-biru, "tangan"
biru) lalu dilabeli ground truth. Diukur:
  - mask accuracy / precision / recall   (piksel mask vs piksel biru GT)
  - detection rate hand                  (berapa % frame tangan terdeteksi)
  - box press accuracy                   (GAS/REM benar-tertekan vs ground truth)
  - center localization error            (jarak pusat deteksi vs GT, rasio)

Output: persentase untuk dimasukkan ke laporan CV (cv_scoring.py).
"""

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main"))
from color_control import (  # noqa: E402
    BLUE_LOWER, BLUE_UPPER, GAS_BOX, BRAKE_BOX,
    make_mask, find_hand_centers, box_is_pressed,
)

W, H = 640, 480
N_FRAMES = 60  # 20 normal, 20 gas, 20 rem


def hsv_jitter(v, mag=8):
    return tuple(int(max(0, min(255, c + np.random.randint(-mag, mag + 1)))) for c in v)


def render_frame(blue_hands):
    """Buat frame sintetis. blue_hands: dict {'gas': (cx,cy)|None, 'brake': ...}."""
    # latar ruangan: warna hangat + noise (bukan biru, agak menantang)
    frame = np.full((H, W, 3), (60, 90, 120), dtype=np.uint8)  # abu hangat
    cv2.rectangle(frame, (0, int(H * 0.7)), (W, H), (40, 70, 95), -1)  # lantai
    # distraktor: objek ungu/cyan pudar mendekati batas range biru
    cv2.circle(frame, (int(W * 0.5), int(H * 0.35)), 50, (180, 120, 60), -1)  # ungu
    cv2.rectangle(frame, (int(W * 0.3), 10), (int(W * 0.45), 90), (200, 180, 60), -1)
    noise = np.random.randint(-12, 12, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    gt_centers = {}
    for name, box in (("gas", GAS_BOX), ("brake", BRAKE_BOX)):
        pos = blue_hands.get(name)
        if pos is None:
            continue
        bx, by, bw, bh = box
        cx = int((bx + bw / 2) * W + np.random.randint(-14, 15))
        cy = int((by + bh / 2) * H + np.random.randint(-12, 13))
        r = int(34 + np.random.randint(-6, 7))
        # "tangan": warna biru di dalam range, sedikit variasi per frame
        col = hsv_jitter((105, 200, 220))
        bgr = cv2.cvtColor(np.uint8([[col]]), cv2.COLOR_HSV2BGR)[0][0]
        cv2.circle(frame, (cx, cy), r, tuple(int(c) for c in bgr), -1)
        cv2.circle(frame, (cx - r // 3, cy - r // 4), r // 2,
                   tuple(int(c) for c in bgr), -1)
        gt_centers[name] = (cx / W, cy / H)
    return frame, gt_centers


def pixel_metrics(mask, frame):
    """Mask piksel vs ground truth 'piksel benar-benar biru range HSV'."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gt = cv2.inRange(hsv, np.array(BLUE_LOWER), np.array(BLUE_UPPER))
    gt = (gt > 0).astype(np.uint8)
    pred = (mask > 0).astype(np.uint8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    acc = (tp + (gt.size - tp - fp - fn)) / gt.size
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, prec, rec, f1


def main():
    np.random.seed(42)
    px_acc = px_prec = px_rec = px_f1 = 0.0
    det_hits = det_total = 0
    press_hits = press_total = 0
    center_err = []
    false_pos = 0  # frame tanpa tangan tapi GAS/REM menyala

    for i in range(N_FRAMES):
        blue = {}
        if i % 3 == 1:
            blue["gas"] = True
        elif i % 3 == 2:
            blue["brake"] = True
        frame, gt = render_frame(blue)

        mask = make_mask(frame)
        a, p, r, f = pixel_metrics(mask, frame)
        px_acc += a; px_prec += p; px_rec += r; px_f1 += f

        centers = find_hand_centers(mask)
        gas_on = box_is_pressed(centers, GAS_BOX)
        brake_on = box_is_pressed(centers, BRAKE_BOX)

        det_total += len(gt)
        for name in gt:
            gx, gy = gt[name]
            matched = any(math.hypot(cx - gx, cy - gy) < 0.08 for cx, cy in centers)
            det_hits += matched
            if matched:
                # error lokalisasi terhadap center terdekat
                d = min(math.hypot(cx - gx, cy - gy) for cx, cy in centers)
                center_err.append(d)

        press_total += 2  # gas + brake per frame
        press_hits += int(gas_on == ("gas" in gt)) + int(brake_on == ("brake" in gt))
        if not gt and (gas_on or brake_on):
            false_pos += 1

    n = N_FRAMES
    result = {
        "frames": n,
        "mask_accuracy_pct": round(px_acc / n * 100, 1),
        "mask_precision_pct": round(px_prec / n * 100, 1),
        "mask_recall_pct": round(px_rec / n * 100, 1),
        "mask_f1_pct": round(px_f1 / n * 100, 1),
        "hand_detection_rate_pct": round(det_hits / max(1, det_total) * 100, 1),
        "box_press_accuracy_pct": round(press_hits / max(1, press_total) * 100, 1),
        "false_positive_rate_pct": round(false_pos / n * 100, 1),
        "center_error_px": round(float(np.mean(center_err)) * W, 1) if center_err else None,
    }
    # skor gabungan deteksi (rata-rata metrik kunci)
    result["detection_score_pct"] = round(
        (
            result["mask_accuracy_pct"] * 0.2
            + result["mask_f1_pct"] * 0.25
            + result["hand_detection_rate_pct"] * 0.25
            + result["box_press_accuracy_pct"] * 0.3
        ), 1
    )
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
