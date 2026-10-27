import json
import math
import random
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_control import (
    BRAKE_BOX,
    CAP_H,
    CAP_W,
    GAS_BOX,
    BLUE_LOWER,
    BLUE_UPPER,
    box_is_pressed,
    find_hand_centers,
    make_mask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_PATH = PROJECT_ROOT / "savegame.json"
ASSETS_DIR = PROJECT_ROOT / "assets"


def _load_img(name, alpha=True):
    """Muat PNG dari assets/; kembalikan None kalau file tidak ada/gagal."""
    try:
        img = pygame.image.load(ASSETS_DIR / name)
        return img.convert_alpha() if alpha else img.convert()
    except Exception:
        return None


W, H = 960, 540
FPS = 60
PX_PER_M = 30.0
STEP = 8
G = 1800.0
ENGINE = 1550.0
BRAKE_DEC = 3000.0
ROLL_RESIST = 90.0
MAX_SPEED = 640.0
WHEEL_R = 13.0
WHEELBASE = 76.0
REST = 0.0
SUSP_TRAVEL = 6.0
SUSP_RATE = 14.0
AIR_TORQUE = 9.5
AIR_SPIN_CAP = 8.5
BOUNCE_VN = -240.0
BOUNCE_REST = 0.25
HEAD_OFF = 24.0
FUEL_TIME = 18.0
MAX_SLOPE = 0.85
MAPS = {
    "countryside": {
        "label": "Countryside (mudah)",
        "seed": 7,
        "rough": 1.0,
        "length_m": 2000,
        "waves": ((40, 600, 1200), (70, 1000, 2000), (22, 300, 600)),
    },
    "cliff": {
        "label": "Cliff (curam)",
        "seed": 21,
        "rough": 1.35,
        "length_m": 2000,
        "waves": ((55, 700, 1200), (95, 1100, 1900), (30, 350, 650)),
    },
}
DEFAULT_SAVE = {
    "settings": {"mirror": True, "camera_cursor": True},
    "maps": {},
}


def load_savegame(path=SAVE_PATH):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    save = {
        "settings": {**DEFAULT_SAVE["settings"], **data.get("settings", {})},
        "maps": dict(data.get("maps", {})),
    }
    return save


def save_savegame(save, path=SAVE_PATH):
    Path(path).write_text(json.dumps(save, indent=2), encoding="utf-8")


def update_records(save, map_id, score, distance):
    rec = save["maps"].get(map_id, {})
    new_record = score > rec.get("best_score", 0)
    rec["best_score"] = max(rec.get("best_score", 0), score)
    rec["best_distance"] = max(rec.get("best_distance", 0.0), distance)
    save["maps"][map_id] = rec
    return new_record


def make_terrain(cfg):
    rng = random.Random(cfg["seed"])
    length_px = int(cfg["length_m"] * PX_PER_M)
    comps = []
    for amp, lam_min, lam_max in cfg["waves"]:
        lam = rng.uniform(lam_min, lam_max)
        ph = rng.uniform(0, math.tau)
        comps.append((amp * cfg["rough"], lam, ph))
    base = 320.0
    raw = []
    for i in range(length_px // STEP + 2):
        x = i * STEP
        y = base
        for amp, lam, ph in comps:
            y -= amp * math.sin((x / lam) * math.tau + ph)
        raw.append(y)
    lim = MAX_SLOPE * STEP
    for i in range(1, len(raw)):
        dy = raw[i] - raw[i - 1]
        if dy > lim:
            raw[i] = raw[i - 1] + lim
        elif dy < -lim:
            raw[i] = raw[i - 1] - lim
    for _ in range(2):
        for i in range(1, len(raw) - 1):
            raw[i] = (raw[i - 1] + 2 * raw[i] + raw[i + 1]) / 4
    pts = []
    for i, y in enumerate(raw):
        x = i * STEP
        t = min(1.0, max(0.0, (x - 250.0) / 700.0))
        pts.append((x, base + (y - base) * t))
    return pts


def ground_y(pts, x):
    if x <= 0:
        return pts[0][1]
    i = int(x // STEP)
    if i >= len(pts) - 1:
        return pts[-1][1]
    x0, y0 = pts[i]
    t = (x - x0) / STEP
    return y0 + (pts[i + 1][1] - y0) * t


def ground_tangent(pts, x):
    i = max(0, min(len(pts) - 2, int(x // STEP)))
    x0, y0 = pts[i]
    x1, y1 = pts[i + 1]
    return x1 - x0, y1 - y0


def make_coins(pts, cfg):
    rng = random.Random(cfg["seed"] * 13 + 5)
    coins = []
    x = 20 * PX_PER_M
    end = pts[-1][0] - 20 * PX_PER_M
    while x < end:
        coins.append(
            {"x": x, "y": ground_y(pts, x) - 60 - rng.uniform(0, 50), "taken": False}
        )
        x += rng.uniform(10, 22) * PX_PER_M
    return coins


def make_fuels(pts, cfg):
    rng = random.Random(cfg["seed"] * 29 + 11)
    fuels = []
    x = 350 * PX_PER_M
    end = pts[-1][0] - 15 * PX_PER_M
    while x < end:
        fuels.append({"x": x, "y": ground_y(pts, x) - 55.0, "taken": False})
        x += rng.uniform(320, 380) * PX_PER_M
    return fuels


def make_car(start_x, pts):
    y = ground_y(pts, start_x) - WHEEL_R - 2
    return {
        "rear": {
            "x": start_x - WHEELBASE / 2,
            "y": y,
            "vx": 0.0,
            "vy": 0.0,
            "grounded": False,
            "roll": 0.0,
        },
        "front": {
            "x": start_x + WHEELBASE / 2,
            "y": y,
            "vx": 0.0,
            "vy": 0.0,
            "grounded": False,
            "roll": 0.0,
        },
        "omega": 0.0,
        "susp": 0.0,
        "start_x": start_x,
        "max_dist": 0.0,
        "coins": 0,
        "fuel": 1.0,
    }


def car_pose(car):
    rw, fw = car["rear"], car["front"]
    a = math.atan2(fw["y"] - rw["y"], fw["x"] - rw["x"])
    return (rw["x"] + fw["x"]) / 2, (rw["y"] + fw["y"]) / 2, a


def step_car(car, pts, gas, brake, dt):
    rw, fw = car["rear"], car["front"]
    for w in (rw, fw):
        gy = ground_y(pts, w["x"])
        w["grounded"] = w["y"] + WHEEL_R >= gy - 0.5
        tx, ty = ground_tangent(pts, w["x"])
        n = math.hypot(tx, ty) or 1.0
        w["tx"], w["ty"] = tx / n, ty / n
        w["nx"], w["ny"] = w["ty"], -w["tx"]
    for w in (rw, fw):
        if w["grounded"]:
            tx, ty, nx, ny = w["tx"], w["ty"], w["nx"], w["ny"]
            sp = w["vx"] * tx + w["vy"] * ty
            if gas:
                sp += ENGINE * dt
            elif brake:
                s = 1.0 if sp > 0 else -1.0
                sp -= min(abs(sp), BRAKE_DEC * dt) * s
            else:
                s = 1.0 if sp > 0 else -1.0
                sp -= min(abs(sp), ROLL_RESIST * dt) * s
            sp = max(-MAX_SPEED * 0.5, min(MAX_SPEED, sp))
            vn = w["vx"] * nx + w["vy"] * ny
            w["vx"] = tx * sp + nx * vn
            w["vy"] = ty * sp + ny * vn
        w["vy"] += G * dt
    for w in (rw, fw):
        w["x"] += w["vx"] * dt
        w["y"] += w["vy"] * dt
        if w["grounded"]:
            sp = w["vx"] * w["tx"] + w["vy"] * w["ty"]
        else:
            sp = w["vx"]
        w["roll"] += (sp / WHEEL_R) * dt * 180.0 / math.pi
    for _ in range(4):
        dx, dy = fw["x"] - rw["x"], fw["y"] - rw["y"]
        dist = math.hypot(dx, dy) or 1e-6
        diff = (dist - WHEELBASE) / dist * 0.5
        rw["x"] += dx * diff
        rw["y"] += dy * diff
        fw["x"] -= dx * diff
        fw["y"] -= dy * diff
    grounded_ws = [w for w in (rw, fw) if w["grounded"]]
    if len(grounded_ws) == 2:
        avg_vx = (rw["vx"] + fw["vx"]) / 2
        avg_vy = (rw["vy"] + fw["vy"]) / 2
        for w in grounded_ws:
            w["vx"], w["vy"] = avg_vx, avg_vy
    for w in (rw, fw):
        gy = ground_y(pts, w["x"])
        pen = w["y"] + WHEEL_R - gy
        if pen > 0:
            w["y"] -= pen
            tx, ty = ground_tangent(pts, w["x"])
            n = math.hypot(tx, ty) or 1.0
            tx, ty = tx / n, ty / n
            nx, ny = ty, -tx
            vn = w["vx"] * nx + w["vy"] * ny
            if vn < 0:
                vt = w["vx"] * tx + w["vy"] * ty
                rest = BOUNCE_REST if vn < BOUNCE_VN else 0.0
                w["vx"] = tx * vt - nx * vn * rest
                w["vy"] = ty * vt - ny * vn * rest
            w["grounded"] = True
    if not (rw["grounded"] or fw["grounded"]):
        if gas and car["omega"] > -AIR_SPIN_CAP:
            car["omega"] -= AIR_TORQUE * dt
        if brake and car["omega"] < AIR_SPIN_CAP:
            car["omega"] += AIR_TORQUE * dt
        cx, cy = (rw["x"] + fw["x"]) / 2, (rw["y"] + fw["y"]) / 2
        ang = car["omega"] * dt
        ca, sa = math.cos(ang), math.sin(ang)
        for w in (rw, fw):
            rx, ry = w["x"] - cx, w["y"] - cy
            w["x"] = cx + rx * ca - ry * sa
            w["y"] = cy + rx * sa + ry * ca
            w["vx"] += -car["omega"] * ry * dt
            w["vy"] += car["omega"] * rx * dt
    else:
        car["omega"] *= max(0.0, 1.0 - 12.0 * dt)
    body_y = (rw["y"] + fw["y"]) / 2.0
    gy_avg = ground_y(pts, (rw["x"] + fw["x"]) / 2.0)
    clearance = max(0.0, gy_avg - body_y - WHEEL_R)
    target_susp = min(SUSP_TRAVEL, clearance * 0.5)
    car["susp"] += (target_susp - car["susp"]) * min(1.0, SUSP_RATE * dt)


def check_death(pts, car):
    rw, fw = car["rear"], car["front"]
    cx, cy, a = car_pose(car)
    hx = cx + math.sin(a) * HEAD_OFF
    hy = cy - math.cos(a) * HEAD_OFF
    if hy >= ground_y(pts, hx) - 3.0 or cy >= ground_y(pts, cx) + 60.0:
        snap = (
            max(
                0.0,
                cy - (ground_y(pts, cx) - 66.0),
                hy - (ground_y(pts, hx) - 9.0),
            )
            + 26.0
        )
        for w in (rw, fw):
            w["y"] -= snap
            w["grounded"] = False
        return True
    return False


class CameraGrab:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.available = self.cap.isOpened()
        if self.available:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.mirror = True
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            if self.available:
                ok, f = self.cap.read()
                if ok:
                    if self.mirror:
                        f = cv2.flip(f, 1)
                    self.frame = cv2.resize(f, (CAP_W, CAP_H))
            else:
                time.sleep(0.05)

    def close(self):
        self._stop.set()
        if self.available:
            self.cap.release()


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Hill Climb Rac")
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.Font(None, 64)
        self.f_menu = pygame.font.Font(None, 34)
        self.f_hud = pygame.font.Font(None, 28)
        self.f_small = pygame.font.Font(None, 22)
        self.camera = CameraGrab()
        self.save = load_savegame()
        self.camera.mirror = self.save["settings"]["mirror"]
        self.running = True
        self.sel = 0
        self.centers = []
        self.gas_cam = False
        self.brake_cam = False
        self.frame_i = 0
        self._dim_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        self._dim_surf.fill((25, 28, 38, 165))
        self.hsv_win = "--no-hsv" not in sys.argv
        self._hsv_created = False
        if "--smoke" in sys.argv:
            pygame.time.set_timer(pygame.QUIT, 2500)
        self._load_assets()
        self.bg = self.assets["home_bg"]

    def _load_assets(self):
        """Muat semua aset dari assets/ (fallback vektor untuk yang None)."""
        bg = _load_img("home_bg.png", alpha=False)
        self.assets = {
            "home_bg": self._scale(bg, (W, H)),
            "map_countryside_bg": self._scale(
                _load_img("map_countryside_bg.png", alpha=False), (W, H)
            ),
            "map_cliff_bg": self._scale(
                _load_img("map_cliff_bg.png", alpha=False), (W, H)
            ),
            "car_body": _load_img("car_body.png"),
            "car_wheel": _load_img("car_wheel.png"),
            "driver": _load_img("driver.png"),
            "coin_frames": [_load_img(f"Coin/image_{i}.png") for i in range(1, 5)],
            "fuel_frames": [_load_img(f"gas/image_{i}.png") for i in range(1, 5)],
        }
        self.assets["car_body_s"] = self._scaled(self.assets["car_body"], 150)
        self.assets["car_wheel_s"] = self._scaled(
            self.assets["car_wheel"], WHEEL_R * 2.0
        )
        driver_s = self._scaled(self.assets["driver"], (HEAD_OFF + 8) * 1.45)
        self.assets["driver_s"] = self._tint(driver_s, (255, 195, 140))
        self.assets["coin_s"] = [
            self._scaled(f, 26) for f in self.assets["coin_frames"] if f is not None
        ]

        self.assets["coin_s"] = self._trim_and_scale(self.assets["coin_frames"], 26)
        self.assets["fuel_s"] = self._trim_and_scale(self.assets["fuel_frames"], 44)

    @staticmethod
    def _trim_and_scale(frames, target_w):
        """Padatkan sprite ke bbox-nya dulu, biar transparansinya nggak ikut diskala."""
        out = []
        for f in frames:
            if f is None:
                continue
            arr = pygame.surfarray.pixels_alpha(f)
            import numpy as _np

            xs = _np.where(arr.max(axis=1) > 10)[0]
            ys = _np.where(arr.max(axis=0) > 10)[0]
            del arr
            if len(xs) == 0 or len(ys) == 0:
                continue
            box = pygame.Rect(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
            trimmed = f.subsurface(box).copy()
            th = int(trimmed.get_height() * target_w / trimmed.get_width())
            out.append(pygame.transform.smoothscale(trimmed, (int(target_w), th)))
        return out

    @staticmethod
    def _scaled(img, target_w):
        if img is None:
            return None
        w, h = img.get_size()
        return pygame.transform.smoothscale(img, (int(target_w), int(h * target_w / w)))

    @staticmethod
    def _tint(img, color):
        if img is None:
            return None
        darkened = img.copy()
        darkened.fill((90, 90, 90), special_flags=pygame.BLEND_RGB_MULT)
        tinted = darkened.copy()
        add = tuple(int(c * 165 / 255) for c in color)
        tinted.fill(add, special_flags=pygame.BLEND_RGB_ADD)
        return tinted

    @staticmethod
    def _scale(img, size):
        if img is None:
            return None
        if img.get_size() == size:
            return img
        return pygame.transform.smoothscale(img, size)

    def close(self):
        self.camera.close()
        if self.hsv_win:
            cv2.destroyAllWindows()
        pygame.quit()

    def _detect_tick(self):
        self.frame_i += 1
        if self.frame_i % 2:
            return
        frame = self.camera.frame
        if frame is None:
            self.centers, self.gas_cam, self.brake_cam = [], False, False
            return
        mask = make_mask(frame)
        self.centers = find_hand_centers(mask)
        self.gas_cam = box_is_pressed(self.centers, GAS_BOX)
        self.brake_cam = box_is_pressed(self.centers, BRAKE_BOX)
        if self.hsv_win:
            self._show_hsv_window(frame, mask)

    HSV_WIN_NAME = "Hill Climb Rac - HSV Tuning"

    def _show_hsv_window(self, frame, mask):
        """Jendela tuning terpisah: frame kamera, mask hitam-putih, dan nilai HSV."""
        if (
            self._hsv_created
            and cv2.getWindowProperty(self.HSV_WIN_NAME, cv2.WND_PROP_VISIBLE) < 1
        ):
            self.hsv_win = False
            return
        view = frame.copy()
        for cx, cy in self.centers:
            cv2.circle(view, (int(cx * CAP_W), int(cy * CAP_H)), 6, (0, 255, 255), -1)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        bar = np.zeros((36, CAP_W, 3), dtype=np.uint8)
        cv2.putText(
            bar,
            f"HSV {BLUE_LOWER}-{BLUE_UPPER} | GAS {'ON' if self.gas_cam else 'off'} | REM {'ON' if self.brake_cam else 'off'} | tekan Q untuk tutup",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panel = np.vstack([view, mask3, bar])
        try:
            cv2.imshow(self.HSV_WIN_NAME, panel)
            self._hsv_created = True
        except cv2.error:
            self.hsv_win = False
            return
        try:
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            key = 0
        if key == ord("q"):
            self.hsv_win = False
            cv2.destroyWindow(self.HSV_WIN_NAME)

    def _text(self, surf, font, txt, pos, color=(255, 255, 255), center=False):
        img = font.render(txt, True, color)
        surf.blit(img, img.get_rect(center=pos) if center else pos)

    def _draw_virtual_cursor(self, surf):
        if not self.save["settings"]["camera_cursor"]:
            return
        col_active = (90, 220, 120) if self.gas_cam else (90, 160, 220)
        base_r = 22 + (4 if (self.gas_cam or self.brake_cam) else 0)
        for gx, gy in self.centers:
            x, y = int(gx * W), int(gy * H)
            tgt = (W - 100, H - 65) if gx >= 0.5 else (100, H - 65)
            pygame.draw.line(surf, (255, 255, 255), (x, y), tgt, 1)
            pygame.draw.line(surf, col_active, (x, y), tgt, 2)
            pygame.draw.circle(surf, col_active, (x, y), base_r, 3)
            pygame.draw.circle(surf, (255, 255, 255), (x, y), 9)
            ph = (pygame.time.get_ticks() % 700) / 700.0
            pr = int(base_r + 8 + ph * 14)
            alpha = int(180 * (1 - ph))
            pulse = pygame.Surface((pr * 2, pr * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                pulse,
                (col_active[0], col_active[1], col_active[2], alpha),
                (pr, pr),
                pr,
                2,
            )
            surf.blit(pulse, (x - pr, y - pr))
            if self.gas_cam or self.brake_cam:
                label = "GAS" if self.gas_cam else "REM"
                img = self.f_small.render(label, True, col_active)
                surf.blit(img, img.get_rect(midtop=(x, y + base_r + 6)))

    def _menu_screen(self, title, labels, subtitle=None, note=None, bg=None):
        bw, bh, gap = 380, 54, 14
        sub_h = (len(subtitle) * 26 + 16) if subtitle else 16
        top, bottom = 146 + sub_h, H - 56
        block = len(labels) * bh + (len(labels) - 1) * gap
        y0 = top + max(0, (bottom - top - block) // 2)
        rects = [
            pygame.Rect(W // 2 - bw // 2, y0 + i * (bh + gap), bw, bh)
            for i in range(len(labels))
        ]
        DWELL_MS, COOLDOWN_MS = 1300, 900
        dwell_i, dwell_t0, cooldown_t0 = None, 0, -COOLDOWN_MS
        while self.running:
            self._detect_tick()
            events = pygame.event.get()
            chosen = None
            mouse = pygame.mouse.get_pos()
            keydowns = [e for e in events if e.type == pygame.KEYDOWN]
            for e in events:
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.MOUSEMOTION:
                    for i, r in enumerate(rects):
                        if r.collidepoint(mouse):
                            self.sel = i
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for i, r in enumerate(rects):
                        if r.collidepoint(mouse):
                            chosen = i
            for e in keydowns:
                if e.key in (pygame.K_DOWN, pygame.K_s):
                    self.sel = (self.sel + 1) % len(labels)
                elif e.key in (pygame.K_UP, pygame.K_w):
                    self.sel = (self.sel - 1) % len(labels)
                elif e.key in (pygame.K_RETURN, pygame.K_SPACE):
                    chosen = self.sel
                elif e.key == pygame.K_ESCAPE:
                    return "ESC"
            if chosen is None:
                now = pygame.time.get_ticks()
                hover = None
                for gx, gy in self.centers[:1]:
                    px, py = gx * W, gy * H
                    for i, r in enumerate(rects):
                        if r.collidepoint(px, py):
                            hover = i
                            break
                if hover is None or hover != dwell_i or now - cooldown_t0 < COOLDOWN_MS:
                    dwell_i, dwell_t0 = hover, now
                elif now - dwell_t0 >= DWELL_MS:
                    chosen = hover
                    cooldown_t0 = now
                    dwell_i = None
                if hover is not None:
                    self.sel = hover
            if chosen is not None:
                self.sel = chosen
                return labels[chosen]
            if bg is not None:
                self.screen.blit(bg, (0, 0))
                self.screen.blit(self._dim_surf, (0, 0))
            else:
                self.screen.fill((25, 28, 38))
            self._text(
                self.screen, self.f_title, title, (W // 2, 100), (255, 210, 60), True
            )
            if subtitle:
                for i, line in enumerate(subtitle):
                    self._text(
                        self.screen,
                        self.f_small,
                        line,
                        (W // 2, 150 + i * 26),
                        (170, 180, 205),
                        True,
                    )
            for i, (r, lab) in enumerate(zip(rects, labels)):
                col = (70, 120, 210) if i == self.sel else (45, 55, 75)
                pygame.draw.rect(self.screen, col, r, border_radius=10)
                pygame.draw.rect(self.screen, (150, 190, 255), r, 2, border_radius=10)
                self._text(
                    self.screen, self.f_menu, lab, r.center, (255, 255, 255), True
                )
            if (
                dwell_i is not None
                and 0 <= dwell_i < len(rects)
                and pygame.time.get_ticks() - cooldown_t0 >= COOLDOWN_MS
            ):
                r0 = rects[dwell_i]
                frac = min(1.0, (pygame.time.get_ticks() - dwell_t0) / DWELL_MS)
                gw, gh = int(r0.w * frac), int(r0.h * frac)
                if gw > 0 and gh > 0:
                    grow = pygame.Rect(0, 0, gw, gh)
                    grow.center = r0.center
                    pygame.draw.rect(
                        self.screen,
                        (70, 200, 90),
                        grow,
                        border_radius=max(2, int(10 * frac)),
                    )
                    self._text(
                        self.screen,
                        self.f_menu,
                        labels[dwell_i],
                        r0.center,
                        (255, 255, 255),
                        True,
                    )
            if note:
                self._text(
                    self.screen,
                    self.f_small,
                    note,
                    (W // 2, H - 24),
                    (150, 160, 180),
                    True,
                )
            self._draw_virtual_cursor(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)
        return "QUIT"

    def screen_home(self):
        ch = self._menu_screen(
            "HILL CLIMB RAC",
            ["Play", "Settings", "Quit"],
            bg=self.bg,
        )
        if ch == "Play":
            return "map_select"
        if ch == "Settings":
            return "settings"
        if ch == "Quit" or ch == "QUIT":
            self.running = False
        return "home"

    def screen_settings(self):
        while self.running:
            s = self.save["settings"]
            labels = [
                f"Mirror Kamera: {'ON' if s['mirror'] else 'OFF'}",
                f"Kursor Deteksi: {'ON' if s['camera_cursor'] else 'OFF'}",
                "Kembali",
            ]
            ch = self._menu_screen(
                "Settings",
                labels,
                subtitle=[
                    "Mirror: balik kiri-kanan kamera",
                    "Kursor: penanda deteksi di layar game",
                ],
                bg=self.bg,
            )
            if ch == labels[0]:
                s["mirror"] = not s["mirror"]
                self.camera.mirror = s["mirror"]
                save_savegame(self.save)
            elif ch == labels[1]:
                s["camera_cursor"] = not s["camera_cursor"]
                save_savegame(self.save)
            else:
                return "home"
        return "QUIT"

    def screen_map_select(self):
        while self.running:
            ch = self._menu_screen(
                "PILIH MAP",
                [MAPS["countryside"]["label"], MAPS["cliff"]["label"], "Kembali"],
                subtitle=[self._best_line("countryside"), self._best_line("cliff")],
                bg=self.bg,
            )
            if ch == MAPS["countryside"]["label"]:
                return "countryside"
            if ch == MAPS["cliff"]["label"]:
                return "cliff"
            return "home"
        return "QUIT"

    def _best_line(self, map_id):
        rec = self.save["maps"].get(map_id, {})
        return f"{MAPS[map_id]['label']}: skor terbaik {rec.get('best_score', 0)} - {rec.get('best_distance', 0.0):.0f} m"

    def screen_play(self, map_id):
        cfg = MAPS[map_id]
        self.cur_map = map_id
        pts = make_terrain(cfg)
        car = make_car(300, pts)
        coins = make_coins(pts, cfg)
        fuels = make_fuels(pts, cfg)
        cam_x, cam_y = 0.0, ground_y(pts, 300) - H * 0.55
        dead = finished = False
        out_of_fuel = False
        over_timer = 0.0
        paused = False
        self.frame_i = 0
        best = self.save["maps"].get(map_id, {}).get("best_score", 0)
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 1 / 30.0)
            self._detect_tick()
            keys = pygame.key.get_pressed()
            gas = (
                self.gas_cam
                or keys[pygame.K_RIGHT]
                or keys[pygame.K_UP]
                or keys[pygame.K_d]
            )
            brake = (
                self.brake_cam
                or keys[pygame.K_LEFT]
                or keys[pygame.K_DOWN]
                or keys[pygame.K_a]
            )
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    paused = not paused
            if paused:
                choice = self._pause_menu()
                if choice == "home":
                    return "home"
                paused = False
                continue
            if not (dead or finished):
                gas_on = gas and car["fuel"] > 0.0
                for _ in range(2):
                    step_car(car, pts, gas_on, brake, dt / 2)
                cx, cy, _ = car_pose(car)
                if gas_on:
                    car["fuel"] -= dt / FUEL_TIME
                if car["fuel"] <= 0.0:
                    car["fuel"] = 0.0
                    if not dead:
                        dead = True
                        out_of_fuel = True
                        cx, cy, _ = car_pose(car)
                        for w in (car["rear"], car["front"]):
                            w["vx"] = w["vy"] = 0.0
                        car["omega"] = 0.0
                        over_timer = -1.2
                if check_death(pts, car):
                    dead = True
                    cx, cy, _ = car_pose(car)
                car["max_dist"] = max(car["max_dist"], (cx - car["start_x"]) / PX_PER_M)
                for c in coins:
                    if not c["taken"] and math.hypot(cx - c["x"], cy - c["y"]) < 36:
                        c["taken"] = True
                        car["coins"] += 1
                for f in fuels:
                    if not f["taken"] and math.hypot(cx - f["x"], cy - f["y"]) < 46:
                        f["taken"] = True
                        car["fuel"] = 1.0
                        if out_of_fuel:
                            dead = False
                            out_of_fuel = False
                            over_timer = 0.0
                if dead:
                    car["vx"] = car["vy"] = car["omega"] = 0.0
                    for w in (car["rear"], car["front"]):
                        w["vx"] = w["vy"] = 0.0
                elif cx >= pts[-1][0] - 150:
                    finished = True
            else:
                over_timer += dt
                if over_timer > 1.0:
                    return self.screen_result(map_id, car, finished, out_of_fuel)
            cam_x += ((cx - W * 0.35) - cam_x) * min(1.0, 8 * dt)
            cam_y += ((cy - H * 0.45) - cam_y) * min(1.0, 8 * dt)
            self._draw_world(pts, coins, fuels, car, cam_x, cam_y)
            self._draw_hud(map_id, car, gas, brake, best, dead, finished, out_of_fuel)
            self._draw_virtual_cursor(self.screen)
            pygame.display.flip()
        return "QUIT"

    def _pause_menu(self):
        while self.running:
            ch = self._menu_screen("JEDA", ["Lanjut", "Home"], note="ESC = lanjut")
            if ch in ("Lanjut", "ESC"):
                return "lanjut"
            if ch == "Home":
                return "home"
            if ch == "QUIT":
                self.running = False
                return "home"
        return "home"

    def screen_result(self, map_id, car, finished, out_of_fuel=False):
        score = int(car["max_dist"]) * 10 + car["coins"] * 50
        new_rec = update_records(self.save, map_id, score, car["max_dist"])
        save_savegame(self.save)
        m = self.save["maps"][map_id]
        stats = [
            f"Jarak: {car['max_dist']:.1f} m   •   Koin: {car['coins']}",
            f"Skor: {score}",
            f"Terbaik: {m['best_score']} skor / {m['best_distance']:.1f} m",
        ]
        if new_rec:
            stats.append("REKOR BARU!")
        while self.running:
            ch = self._menu_screen(
                "FINISH!" if finished else "BENSIN HABIS!",
                ["Main Lagi", "Pilih Map", "Home"],
                subtitle=stats,
                note=None,
                bg=self.bg,
            )
            if ch == "Main Lagi":
                return "play"
            if ch == "Pilih Map" or ch == "ESC":
                return "map_select"
            if ch == "Home":
                return "home"
            if ch == "QUIT":
                self.running = False
                return "home"
        return "home"

    def _draw_world(self, pts, coins, fuels, car, cam_x, cam_y):
        map_id = getattr(self, "cur_map", "countryside")
        bg = self.assets.get(f"map_{map_id}_bg")
        if bg is not None:
            bw = bg.get_width()
            off = int(-cam_x * 0.25) % bw
            for i in range(int(W / bw) + 2):
                self.screen.blit(bg, (off + i * bw - bw, 0))
        else:
            self.screen.fill((125, 185, 235))
        surf_pts = [(0, H)]
        for sx in range(-STEP, W + STEP + 1, STEP):
            wx = cam_x + sx
            surf_pts.append((sx, ground_y(pts, wx) - cam_y))
        surf_pts.append((W, H))
        if len(surf_pts) > 2:
            pygame.draw.polygon(self.screen, (96, 70, 50), surf_pts)
            if f"speckles_{map_id}" not in self.assets:
                rng = random.Random("spek_" + map_id)
                base = (96, 70, 50)
                self.assets[f"speckles_{map_id}"] = [
                    (
                        rng.uniform(0, pts[-1][0]),
                        rng.uniform(16, 280),
                        rng.randint(2, 5),
                        tuple(c + rng.randint(-16, 14) for c in base),
                    )
                    for _ in range(300)
                ]
            for wx, dy, r, col in self.assets[f"speckles_{map_id}"]:
                sx = wx - cam_x
                if -8 <= sx <= W + 8:
                    gy = ground_y(pts, wx) + dy - cam_y
                    if gy < H + 8:
                        pygame.draw.circle(self.screen, col, (int(sx), int(gy)), r)
            fringe = (86, 160, 70) if map_id == "countryside" else (150, 150, 160)
            hi = (118, 190, 92) if map_id == "countryside" else (185, 185, 195)
            pygame.draw.lines(self.screen, fringe, False, surf_pts[1:-1], 9)
            pts_hi = [(x, y - 3) for x, y in surf_pts[1:-1]]
            pygame.draw.lines(self.screen, hi, False, pts_hi, 3)
        frame = getattr(self, "frame_i", 0)
        coin_s = self.assets.get("coin_s") or []
        if coin_s:
            cimg = coin_s[(frame // 8) % len(coin_s)]
            for c in coins:
                if c["taken"]:
                    continue
                sx, sy = c["x"] - cam_x, c["y"] - cam_y
                if -30 <= sx <= W + 30:
                    rect = cimg.get_rect(center=(int(sx), int(sy)))
                    self.screen.blit(cimg, rect.topleft)
        else:
            for c in coins:
                if c["taken"]:
                    continue
                sx, sy = c["x"] - cam_x, c["y"] - cam_y
                if -20 <= sx <= W + 20:
                    pygame.draw.circle(
                        self.screen, (255, 205, 60), (int(sx), int(sy)), 9
                    )
                    pygame.draw.circle(
                        self.screen, (180, 140, 20), (int(sx), int(sy)), 9, 2
                    )
        fuel_s = self.assets.get("fuel_s") or []
        if fuel_s:
            fimg = fuel_s[(frame // 8) % len(fuel_s)]
            for f in fuels:
                if f["taken"]:
                    continue
                sx, sy = f["x"] - cam_x, f["y"] - cam_y
                if -50 <= sx <= W + 50:
                    rect = fimg.get_rect(center=(int(sx), int(sy)))
                    self.screen.blit(fimg, rect.topleft)
        rw, fw = car["rear"], car["front"]
        _, _, a = car_pose(car)
        fwd = (math.cos(a), math.sin(a))
        up = (math.sin(a), -math.cos(a))

        def tp(wx, wy):
            return (int(wx - cam_x), int(wy - cam_y))

        cx, cy, _ = car_pose(car)
        body = self.assets["car_body_s"]
        wheel_img = self.assets["car_wheel_s"]
        driver = self.assets["driver_s"]
        if body is not None and wheel_img is not None:
            spr = pygame.transform.rotate(body, -math.degrees(a))
            r = spr.get_rect(
                center=(
                    int(cx + up[0] * (44.0 + car.get("susp", 0.0)) - cam_x),
                    int(cy + up[1] * (44.0 + car.get("susp", 0.0)) - cam_y),
                )
            )
            self.screen.blit(spr, r.topleft)
            hx, hy = (
                cx + up[0] * (HEAD_OFF + car.get("susp", 0.0)),
                cy + up[1] * (HEAD_OFF + car.get("susp", 0.0)),
            )
            if driver is not None:
                dcx, dcy = (
                    hx + fwd[0] * -31.0 + up[0] * 47.0 - cam_x,
                    hy + fwd[1] * -31.0 + up[1] * 47.0 - cam_y,
                )
                dspr = pygame.transform.rotate(driver, -math.degrees(a))
                dh = dspr.get_rect(center=(int(dcx), int(dcy)))
                self.screen.blit(dspr, dh.topleft)
            else:
                pygame.draw.circle(self.screen, (255, 220, 180), tp(hx, hy), 9)
                pygame.draw.circle(self.screen, (240, 240, 240), tp(hx, hy), 11, 2)
            for w in (rw, fw):
                side = 1.0 if w is fw else -1.0
                ang = -math.degrees(w.get("roll", 0.0))
                wr = pygame.transform.rotate(wheel_img, ang)
                wx_v = cx + fwd[0] * side * 34.0 + up[0] * 14.0
                wy_v = cy + fwd[1] * side * 34.0 + up[1] * 14.0
                wy_v = max(wy_v, ground_y(pts, wx_v) - WHEEL_R)
                rr = wr.get_rect(center=(int(wx_v - cam_x), int(wy_v - cam_y)))
                self.screen.blit(wr, rr.topleft)
        else:
            corners = [
                (rw["x"] + up[0] * 10, rw["y"] + up[1] * 10),
                (
                    rw["x"] + up[0] * 34 + fwd[0] * 10,
                    rw["y"] + up[1] * 34 + fwd[1] * 10,
                ),
                (
                    fw["x"] + up[0] * 34 - fwd[0] * 10,
                    fw["y"] + up[1] * 34 - fwd[1] * 10,
                ),
                (fw["x"] + up[0] * 10, fw["y"] + up[1] * 10),
            ]
            pygame.draw.polygon(self.screen, (200, 45, 45), [tp(*p) for p in corners])
            hx, hy = cx + up[0] * HEAD_OFF, cy + up[1] * HEAD_OFF
            pygame.draw.circle(self.screen, (255, 220, 180), tp(hx, hy), 9)
            pygame.draw.circle(self.screen, (240, 240, 240), tp(hx, hy), 11, 2)
            for w in (rw, fw):
                pygame.draw.circle(
                    self.screen, (30, 30, 30), tp(w["x"], w["y"]), int(WHEEL_R)
                )
                pygame.draw.circle(
                    self.screen, (90, 90, 90), tp(w["x"], w["y"]), int(WHEEL_R), 2
                )

    def _draw_hud(
        self, map_id, car, gas, brake, best, dead, finished, out_of_fuel=False
    ):
        hud = [
            f"Jarak: {car['max_dist']:.0f} m",
            f"Koin: {car['coins']}",
            f"Skor: {int(car['max_dist']) * 10 + car['coins'] * 50}",
            f"Terbaik: {best}",
        ]
        for i, t in enumerate(hud):
            self._text(self.screen, self.f_hud, t, (12, 10 + i * 26))
        fuel = max(0.0, min(1.0, car.get("fuel", 1.0)))
        frect = pygame.Rect(12, 10 + len(hud) * 26 + 6, 140, 14)
        pygame.draw.rect(self.screen, (20, 22, 30), frect)
        if fuel > 0.5:
            fcol = (70, 200, 90)
        elif fuel > 0.25:
            fcol = (250, 200, 60)
        else:
            fcol = (230, 70, 60)
        ffill = frect.copy()
        ffill.width = int(frect.width * fuel)
        pygame.draw.rect(self.screen, fcol, ffill)
        pygame.draw.rect(self.screen, (255, 255, 255), frect, 2)
        self._text(
            self.screen, self.f_small, "Bensin", (frect.right + 8, frect.top - 4)
        )
        if not self.camera.available:
            self._text(
                self.screen,
                self.f_small,
                "Kamera OFF - pakai keyboard (panah)",
                (12, H - 30),
                (255, 200, 120),
            )
        gas_rect = pygame.Rect(W - 170, H - 100, 140, 70)
        brake_rect = pygame.Rect(30, H - 100, 140, 70)
        for rect, label, on in ((gas_rect, "GAS", gas), (brake_rect, "REM", brake)):
            col = (255, 200, 40) if on else (60, 70, 90)
            pygame.draw.rect(self.screen, col, rect, border_radius=12)
            pygame.draw.rect(
                self.screen,
                (255, 255, 255) if on else (120, 130, 150),
                rect,
                3,
                border_radius=12,
            )
            self._text(
                self.screen,
                self.f_menu,
                label,
                rect.center,
                (20, 20, 20) if on else (200, 205, 215),
                True,
            )
        if dead:
            self._text(
                self.screen,
                self.f_title,
                "BENSIN HABIS!" if out_of_fuel else "MOBIL RUSAK!",
                (W // 2, H // 2 - 60),
                (255, 90, 90),
                True,
            )
        elif finished:
            self._text(
                self.screen,
                self.f_title,
                "FINISH!",
                (W // 2, H // 2 - 60),
                (120, 255, 120),
                True,
            )

    def run(self):
        current_map = None
        state = "home"
        while self.running:
            if state == "home":
                state = self.screen_home()
            elif state == "settings":
                state = self.screen_settings()
            elif state == "map_select":
                picked = self.screen_map_select()
                if picked in MAPS:
                    current_map = picked
                    state = "play"
                else:
                    state = picked
            elif state == "play" and current_map:
                state = self.screen_play(current_map)
            else:
                print(f"[main] state tak dikenal: {state!r} - kembali ke home")
                state = "home"


def main():
    game = Game()
    try:
        game.run()
    finally:
        game.close()


if __name__ == "__main__":
    main()
