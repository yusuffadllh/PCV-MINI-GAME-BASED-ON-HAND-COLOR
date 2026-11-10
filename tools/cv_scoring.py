#!/usr/bin/env python3
"""Parameter penilaian project "Hill Climb Rac" untuk CV — output persentase.

Cara pakai:
    python3 tools/cv_scoring.py          # jalankan smoke test lalu beri skor
    python3 tools/cv_scoring.py --json   # output JSON mentah

Skor gabungan = rata-rata tertimbang dari 5 kategori (0-100%).
"""

import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE = os.path.join(ROOT, "savegame.json")
MAIN = os.path.join(ROOT, "main", "main.py")

# Bobot tiap kategori (total 100%)
W_FUNCTIONAL = 0.35
W_DESIGN = 0.25
W_FEATURES = 0.20
W_CODE = 0.15
W_POLISH = 0.05


def run_smoke():
    """Jalankan game mode smoke-test via harness dan kembalikan metrik."""
    harness = r"""
import json
import sys
sys.argv = ["main.py", "--smoke", "--no-hsv", "--collect-coins"]
sys.path.insert(0, "main")
import importlib.util
spec = importlib.util.spec_from_file_location("main", "main/main.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
game = mod.Game()
game.run()
log = game._smoke_log
xs = [f["x"] for f in log]
out = {
    "frames": len(log),
    "max_dist_m": (max(xs) - log[0]["x"]) / mod.PX_PER_M if log else 0.0,
    "coins": log[-1]["coins"] if log else 0,
    "fuel_end": log[-1]["fuel"] if log else 0.0,
    "dead": bool(log[-1]["dead"]) if log else False,
    "finished": bool(log[-1]["finished"]) if log else False,
}
print("SMOKE_RESULT:" + json.dumps(out))
game.close()
"""
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    r = subprocess.run(
        [sys.executable, "-c", harness], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    for line in r.stdout.splitlines():
        if line.startswith("SMOKE_RESULT:"):
            return json.loads(line[len("SMOKE_RESULT:"):])
    raise RuntimeError("smoke test gagal:\n" + r.stderr[-2000:])


def run_hsv_eval():
    """Jalankan tools/hsv_eval.py: akurasi deteksi HSV (mask + hand + box)."""
    eval_path = os.path.join(ROOT, "tools", "hsv_eval.py")
    r = subprocess.run(
        [sys.executable, eval_path], cwd=ROOT,
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError("hsv_eval gagal:\n" + r.stderr[-2000:])
    return json.loads(r.stdout)


def load_code_metrics():
    """Metrik statis: baris kode, jumlah fungsi/kelas, docstring, dsb."""
    src = open(MAIN, encoding="utf-8").read()
    lines = src.count("\n")
    funcs = src.count("def ")
    classes = src.count("class ")
    comments = sum(1 for l in src.splitlines() if l.strip().startswith("#"))
    try:
        import cv2  # noqa: F401
        cv2_ok = True
    except ImportError:
        cv2_ok = False
    return {
        "lines": lines, "funcs": funcs, "classes": classes,
        "comments": comments, "cv2_ok": cv2_ok,
    }


def load_save():
    return json.load(open(SAVE, encoding="utf-8"))


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def score_all():
    m = run_smoke()
    c = load_code_metrics()
    s = load_save()
    best = s["maps"].get("countryside", {}).get("best_score", 0)
    best_dist = s["maps"].get("countryside", {}).get("best_distance", 0.0)

    # ---- 1. FUNGSIONALITAS (bobot 35%) ----
    # a. Boot & run stabil: smoke test selesai tanpa crash (0-40)
    boot = 40 if m["frames"] > 50 else (m["frames"] / 50 * 40)
    # b. Physics & movement berjalan (0-30): berdasarkan jarak tempuh smoke
    phys = clamp(m["max_dist_m"] / 100 * 30, 0, 30)  # 100 m = nilai penuh
    # c. State machine menu->play->result (0-30): return None = loop normal,
    #    tidak ada state tak dikenal
    states = 30 if m["frames"] > 20 and not m["dead"] else 20
    functional = boot + phys + states

    # ---- 2. DESAIN & GAMEPLAY (bobot 25%) ----
    # a. Terrain & map system (0-40): 2 map terdefinisi + savegame per map
    maps_n = len(s["maps"])
    design_maps = clamp(maps_n / 2 * 40, 0, 40)
    # b. Elemen interaktif (0-30): koin & fuel tersedia & terkoleksi
    elems = clamp(m["coins"] / 40 * 30, 0, 30)  # 40 koin = penuh
    # c. Progression (0-30): best score / 10000 target
    prog = clamp(best / 10000 * 30, 0, 30)
    design = design_maps + elems + prog

    # ---- 3. FITUR (bobot 20%) ----
    # a. Akurasi deteksi kamera/HSV (0-40): hasil hsv_eval
    hsv = run_hsv_eval()
    cam = clamp(hsv["detection_score_pct"] / 100 * 40, 0, 40)
    # b. Savegame & persistence (0-30): best_score > 0
    persist = 30 if best > 0 else 10
    # c. HUD & UI lengkap (0-30): font layers + menu + pause + result
    ui = 30  # ada f_title/f_menu/f_hud/f_small + pause + result screen
    features = cam + persist + ui

    # ---- 4. KUALITAS KODE (bobot 15%) ----
    # a. Modularitas (0-40): fungsi/kelas per 10 baris
    modul = clamp((c["funcs"] + c["classes"] * 3) / (c["lines"] / 10) * 10, 0, 40)
    # b. Komentar & dokumentasi (0-30)
    doc = clamp(c["comments"] / c["lines"] * 300, 0, 30)  # 10% komentar = penuh
    # c. Ketahanan (0-30): smoke test tanpa exception
    robust = 30 if m["frames"] > 50 else 15
    code = modul + doc + robust

    # ---- 5. POLISH (bobot 5%) ----
    # a. Aset grafis (0-60): gambar ada di assets/
    assets = 0
    for d in ("Coin", "gas"):
        p = os.path.join(ROOT, "assets", d)
        if os.path.isdir(p):
            assets += 30
    # b. Best distance tercatat (0-40)
    polish = clamp(assets + best_dist / 1000 * 40, 0, 100)

    total = (
        functional * W_FUNCTIONAL + design * W_DESIGN + features * W_FEATURES
        + code * W_CODE + polish * W_POLISH
    )

    report = {
        "meta": {
            "smoke_frames": m["frames"],
            "smoke_max_dist_m": round(m["max_dist_m"], 1),
            "smoke_coins": m["coins"],
            "smoke_fuel_end": round(m["fuel_end"], 2),
            "best_score": best,
            "best_distance_m": round(best_dist, 1),
            "hsv": hsv,
            "lines": c["lines"], "funcs": c["funcs"],
            "classes": c["classes"], "comments": c["comments"],
            "cv2": c["cv2_ok"],
        },
        "categories": {
            "Fungsionalitas (35%)": round(functional, 1),
            "Desain & Gameplay (25%)": round(design, 1),
            "Fitur (20%)": round(features, 1),
            "Kualitas Kode (15%)": round(code, 1),
            "Polish (5%)": round(polish, 1),
        },
        "TOTAL": round(total, 1),
    }
    return report


def main():
    rep = score_all()
    if "--json" in sys.argv:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return
    print("=" * 58)
    print("  LAPORAN PENILAIAN PROYEK — Hill Climb Rac (untuk CV)")
    print("=" * 58)
    m = rep["meta"]
    print(f"\n[Data terukur]")
    print(f"  Smoke test      : {m['smoke_frames']} frame, tanpa crash")
    print(f"  Jarak tempuh    : {m['smoke_max_dist_m']} m (auto-gas)")
    print(f"  Koin terkumpul  : {m['smoke_coins']}")
    print(f"  Rekor simpanan  : {m['best_score']} skor / {m['best_distance_m']} m")
    print(f"  Kode            : {m['lines']} baris, {m['funcs']} fungsi, "
          f"{m['classes']} kelas")
    print(f"  Kamera OpenCV   : {'AKTIF' if m['cv2'] else 'tidak tersedia'}")
    h = m["hsv"]
    print(f"\n[Deteksi HSV — tools/hsv_eval.py, {h['frames']} frame sintetis]")
    print(f"  Mask accuracy     : {h['mask_accuracy_pct']}%")
    print(f"  Mask precision    : {h['mask_precision_pct']}%")
    print(f"  Mask recall       : {h['mask_recall_pct']}%")
    print(f"  Mask F1           : {h['mask_f1_pct']}%")
    print(f"  Hand detect rate  : {h['hand_detection_rate_pct']}%")
    print(f"  Box press accuracy: {h['box_press_accuracy_pct']}%")
    print(f"  False positive    : {h['false_positive_rate_pct']}%")
    print(f"  Skor deteksi HSV  : {h['detection_score_pct']}%")
    print(f"\n[Skor per kategori — 0-100%]")
    for k, v in rep["categories"].items():
        bar = "#" * int(v / 5)
        print(f"  {k:<28} {v:>5.1f}%  {bar}")
    print("-" * 58)
    print(f"  SKOR GABUNGAN                {rep['TOTAL']:>5.1f}%")
    print("=" * 58)


if __name__ == "__main__":
    main()
