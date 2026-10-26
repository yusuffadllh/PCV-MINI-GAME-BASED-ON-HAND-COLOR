import cv2
import numpy as np
import socket

CAMERA_INDEX = 0
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
BLUE_LOWER = (90, 80, 80)
BLUE_UPPER = (130, 255, 255)
GAS_BOX = (0.82, 0.72, 0.18, 0.28)
BRAKE_BOX = (0.00, 0.72, 0.18, 0.28)
MIN_AREA = 2000
MAX_HANDS = 2


def make_mask(frame):
    blur = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(BLUE_LOWER), np.array(BLUE_UPPER))
    mask = cv2.medianBlur(mask, 11)
    return mask


def find_hand_centers(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    contours.sort(key=cv2.contourArea, reverse=True)
    h, w = mask.shape[:2]
    centers = []
    for contour in contours[:MAX_HANDS]:
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        centers.append(((m["m10"] / m["m00"]) / w, (m["m01"] / m["m00"]) / h))
    return centers


def point_in_box(cx_ratio, cy_ratio, box):
    bx, by, bw, bh = box
    return bx <= cx_ratio <= bx + bw and by <= cy_ratio <= by + bh


def box_is_pressed(centers, box):
    return any(point_in_box(cx, cy, box) for cx, cy in centers)


def draw_box(frame, box, color, active):
    """Util pengujian; kotak tidak digambar di jendela kamera."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = box
    p1 = (int(bx * w), int(by * h))
    p2 = (int((bx + bw) * w), int((by + bh) * h))
    cv2.rectangle(frame, p1, p2, color, 4 if active else 1)


def udp_command_for(centers):
    command = ""
    if box_is_pressed(centers, GAS_BOX):
        command += "G"
    if box_is_pressed(centers, BRAKE_BOX):
        command += "B"
    return command


CAP_W, CAP_H = 640, 480
CAP_FPS = 30


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        mask = make_mask(frame)
        centers = find_hand_centers(mask)
        gas_pressed = box_is_pressed(centers, GAS_BOX)
        brake_pressed = box_is_pressed(centers, BRAKE_BOX)
        command = udp_command_for(centers)
        sock.sendto(command.encode(), (UDP_IP, UDP_PORT))
        h, w = frame.shape[:2]
        for cx, cy in centers:
            cv2.circle(frame, (int(cx * w), int(cy * h)), 8, (255, 255, 255), -1)
        cv2.putText(
            frame,
            command or "-",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imshow("vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
