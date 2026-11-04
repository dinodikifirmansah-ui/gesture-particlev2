import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
from collections import deque

W, H, N = 640, 480, 1200
rng = np.random.default_rng()

def shape_stars():
    return np.column_stack([rng.uniform(0, W, N), rng.uniform(0, H, N)])

def shape_heart():
    t = rng.uniform(0, 2*np.pi, N)
    x = 16*np.sin(t)**3
    y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
    pts = np.column_stack([x, -y]) * 11
    return pts + rng.normal(0, 1.2, pts.shape) + [W/2, H/2-10]

def shape_text():
    img = np.zeros((H, W), np.uint8)
    text = "I LOVE YOU"
    scale, thick = 2.2, 6
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    cv2.putText(img, text, ((W-tw)//2, (H+th)//2),
                cv2.FONT_HERSHEY_DUPLEX, scale, 255, thick)
    ys, xs = np.nonzero(img)
    idx = rng.integers(0, len(xs), N)
    return np.column_stack([xs[idx], ys[idx]]).astype(float)

def shape_saturn():
    n_body = int(N*0.6)
    r = 75*np.sqrt(rng.uniform(0, 1, n_body))
    a = rng.uniform(0, 2*np.pi, n_body)
    body = np.column_stack([r*np.cos(a), r*np.sin(a)])
    n_ring = N-n_body
    a = rng.uniform(0, 2*np.pi, n_ring)
    ring = np.column_stack([150*np.cos(a), 38*np.sin(a)])
    tilt = np.deg2rad(-20)
    rot = np.array([[np.cos(tilt), -np.sin(tilt)],
                    [np.sin(tilt), np.cos(tilt)]])
    return np.vstack([body, ring @ rot.T]) + [W/2, H/2]

SHAPES = {
    "stars": (shape_stars, (255, 160, 80)),
    "heart": (shape_heart, (90, 60, 255)),
    "text": (shape_text, (200, 230, 80)),
    "saturn": (shape_saturn, (60, 130, 255)),
}
FINGER_MODE = {5:"stars", 0:"heart", 1:"text", 2:"saturn", 3:"saturn"}

mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, model_complexity=0,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

def count_fingers(lm):
    p = lambda i: np.array([lm[i].x, lm[i].y])
    wrist = p(0)
    count = 0
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        if np.linalg.norm(p(tip)-wrist) > np.linalg.norm(p(pip)-wrist)*1.1:
            count += 1
    if np.linalg.norm(p(4)-p(17)) > np.linalg.norm(p(3)-p(17))*1.05:
        count += 1
    return count

def new_state():
    pos = shape_stars()
    return {
        "pos": pos, "vel": np.zeros_like(pos), "target": pos.copy(),
        "mode": "stars", "last_mode": None, "history": []
    }

def process_frame(frame, state):
    if frame is None:
        return None, state, "Arahkan kamera ke tangan."
    if state is None:
        state = new_state()

    frame = cv2.resize(frame, (W, H))
    frame = cv2.flip(frame, 1)
    result = hand_detector.process(frame)

    mode = state["mode"]
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(
            frame, landmarks, mp_hands.HAND_CONNECTIONS
        )
        state["history"].append(count_fingers(landmarks.landmark))
        state["history"] = state["history"][-6:]
        stable = max(set(state["history"]), key=state["history"].count)
        mode = FINGER_MODE.get(stable, mode)
        status = f"Gestur terdeteksi: {stable} jari"
    else:
        state["history"] = []
        mode = "stars"
        status = "Tangan belum terdeteksi — tampilkan telapak tangan."

    if mode != state["last_mode"]:
        state["target"] = SHAPES[mode][0]()
        state["last_mode"] = mode
    if mode == "stars":
        state["target"] += rng.normal(0, 0.8, state["target"].shape)

    state["vel"] = (state["vel"]*0.85
                    + (state["target"]-state["pos"])*0.06
                    + rng.normal(0, 0.35, state["pos"].shape))
    state["pos"] += state["vel"]

    particles = np.zeros((H, W, 3), np.uint8)
    color = SHAPES[mode][1]
    for x, y in state["pos"].astype(int):
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(particles, (x, y), 1, color, -1)

    # Tampilkan kamera dan partikel berdampingan dalam satu output web.
    combined = np.concatenate([frame, particles], axis=1)
    cv2.putText(combined, "KAMERA", (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,255,255), 2)
    cv2.putText(combined, mode.upper(), (W+18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return combined, state, f"{status} | Efek aktif: {mode.upper()}"

with gr.Blocks(title="Gesture Particle Experience", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ✨ Gesture Particle Experience\n"
        "Kontrol partikel menggunakan gestur tangan melalui kamera browser.\n\n"
        "**5 jari:** bintang · **0 jari:** hati · **1 jari:** I LOVE YOU · "
        "**2–3 jari:** Saturnus"
    )
    with gr.Row():
        camera = gr.Image(
            sources=["webcam"], type="numpy", streaming=True,
            label="Kamera — izinkan akses kamera browser"
        )
    output = gr.Image(label="Tampilan kamera dan partikel", streaming=True)
    status = gr.Markdown("Tekan tombol kamera, lalu arahkan tangan ke kamera.")
    state = gr.State(new_state())
    camera.stream(
        fn=process_frame, inputs=[camera, state],
        outputs=[output, state, status], time_limit=60, stream_every=0.12
    )
    gr.Markdown(
        "Catatan: gunakan HTTPS (Spaces menyediakan HTTPS) dan izinkan akses kamera. "
        "Kinerja bergantung pada perangkat dan koneksi internet."
    )

if __name__ == "__main__":
    demo.queue().launch()
