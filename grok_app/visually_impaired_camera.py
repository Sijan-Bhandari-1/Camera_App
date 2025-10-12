import cv2
import pyttsx3
import speech_recognition as sr
import time
import threading
import queue
import os
import sys
import re

# ------------------------------
# SPEECH ENGINE
# ------------------------------
speech_queue = queue.Queue()

def _init_tts_engine():
    """Initialize pyttsx3 engine with cross-platform driver fallbacks.

    - Windows: sapi5
    - macOS: nsss
    - Linux: espeak/espeak-ng (falls back to default)
    """
    driver_candidates = []
    if sys.platform.startswith("win"):
        driver_candidates = ["sapi5", None]
    elif sys.platform == "darwin":
        driver_candidates = ["nsss", None]
    else:
        # Assume Linux
        driver_candidates = ["espeak", "espeak-ng", None]

    last_error = None
    for driver in driver_candidates:
        try:
            engine = pyttsx3.init(driverName=driver) if driver else pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.setProperty("volume", 1.0)

            # Prefer an English voice if available, otherwise keep default
            try:
                voices = engine.getProperty("voices")
                chosen_voice_id = None
                for v in voices:
                    name_lower = (getattr(v, "name", "") or "").lower()
                    langs = getattr(v, "languages", []) or []
                    if any("en" in str(lang).lower() for lang in langs) or "english" in name_lower:
                        chosen_voice_id = v.id
                        break
                if chosen_voice_id:
                    engine.setProperty("voice", chosen_voice_id)
            except Exception:
                pass

            return engine
        except Exception as e:
            last_error = e
            continue
    # If all drivers fail, re-raise last error
    if last_error:
        raise last_error
    return pyttsx3.init()

engine = _init_tts_engine()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        print(f"[DEBUG] Speaking: {text}")
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("[DEBUG] Speech error:", e)
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)
    time.sleep(0.3)

# ------------------------------
# SPEECH RECOGNITION
# ------------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen(phrase_time_limit: int = 5) -> str:
    # Do not speak while the microphone is open to avoid feedback
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        print("[DEBUG] Listening for voice input...")
        try:
            audio = recognizer.listen(source, phrase_time_limit=phrase_time_limit)
        except Exception as e:
            print("[DEBUG] Audio capture failed:", e)
            return ""
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"[DEBUG] Heard: {text}")
        return text
    except Exception:
        speak("I did not catch that. Please say again.")
        return ""

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _section_label(code: str) -> str:
    return {
        "tl": "Top Left",
        "tr": "Top Right",
        "bl": "Bottom Left",
        "br": "Bottom Right",
        "c": "Center",
    }.get(code, "Center")

def get_voice_choice() -> str:
    speak("Welcome. Please select your target section. The options are: Top Left, Top Right, Bottom Left, Bottom Right, or Center.")
    for attempt in range(3):
        response_raw = listen()
        response = _normalize_text(response_raw)
        if not response:
            continue

        if "top left" in response or "upper left" in response or "left top" in response:
            choice = "tl"
        elif "top right" in response or "upper right" in response or "right top" in response:
            choice = "tr"
        elif "bottom left" in response or "lower left" in response or "left bottom" in response:
            choice = "bl"
        elif "bottom right" in response or "lower right" in response or "right bottom" in response:
            choice = "br"
        elif "center" in response or "centre" in response or "middle" in response:
            choice = "c"
        else:
            speak("I did not catch that. Please say again.")
            continue

        speak(f"Target section selected: {_section_label(choice)}.")
        return choice

    # Fallback to center after several attempts
    speak("Target section selected: Center.")
    return "c"

# ------------------------------
# LAYOUT + DETECTION
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
if eye_cascade.empty():
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def draw_layout(frame, selected):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    box_w, box_h = w // 3, h // 3
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2

    color = (0, 255, 0)
    thickness = 1
    cv2.line(frame, (cx, 0), (cx, y1), color, thickness)
    cv2.line(frame, (cx, y2), (cx, h), color, thickness)
    cv2.line(frame, (0, cy), (x1, cy), color, thickness)
    cv2.line(frame, (x2, cy), (w, cy), color, thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "TL", (10, 30), font, 0.7, color, 2)
    cv2.putText(frame, "TR", (w - 50, 30), font, 0.7, color, 2)
    cv2.putText(frame, "BL", (10, h - 20), font, 0.7, color, 2)
    cv2.putText(frame, "BR", (w - 50, h - 20), font, 0.7, color, 2)

    overlay = frame.copy()
    highlight = (0, 255, 255)
    alpha = 0.25
    if selected == "tl":
        cv2.rectangle(overlay, (0, 0), (cx, cy), highlight, -1)
    elif selected == "tr":
        cv2.rectangle(overlay, (cx, 0), (w, cy), highlight, -1)
    elif selected == "bl":
        cv2.rectangle(overlay, (0, cy), (cx, h), highlight, -1)
    elif selected == "br":
        cv2.rectangle(overlay, (cx, cy), (w, h), highlight, -1)
    elif selected == "c":
        cv2.rectangle(overlay, (x1, y1), (x2, y2), highlight, -1)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame, (x1, y1, x2, y2, cx, cy)

def detect_section(face_cx, face_cy, w, h, x1, y1, x2, y2, cx, cy):
    if x1 <= face_cx <= x2 and y1 <= face_cy <= y2:
        return "c"
    elif face_cx < cx and face_cy < cy:
        return "tl"
    elif face_cx >= cx and face_cy < cy:
        return "tr"
    elif face_cx < cx and face_cy >= cy:
        return "bl"
    else:
        return "br"

def _target_center(selected: str, w: int, h: int, cx: int, cy: int, x1: int, y1: int, x2: int, y2: int):
    if selected == "tl":
        return (cx // 2, cy // 2)
    if selected == "tr":
        return (cx + (w - cx) // 2, cy // 2)
    if selected == "bl":
        return (cx // 2, cy + (h - cy) // 2)
    if selected == "br":
        return (cx + (w - cx) // 2, cy + (h - cy) // 2)
    # center
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def guidance(face_cx: int, face_cy: int, current: str, target: str, dims, layout_rect) -> tuple[str, bool, bool]:
    """Return (instruction, aligned, near).

    - instruction: what to speak next
    - aligned: True when inside target and sufficiently centered
    - near: True when close enough to say "Good, almost there."
    """
    w, h = dims
    x1, y1, x2, y2, cx, cy = layout_rect
    tx, ty = _target_center(target, w, h, cx, cy, x1, y1, x2, y2)
    dx = tx - face_cx
    dy = ty - face_cy
    dist = (dx * dx + dy * dy) ** 0.5

    base_unit = 0.08 * min(w, h)  # guidance sensitivity
    near_unit = 0.04 * min(w, h)

    # Determine alignment: inside target section and close to target center
    in_center_box = (x1 <= face_cx <= x2 and y1 <= face_cy <= y2)
    in_quadrant = (
        (target == "tl" and face_cx < cx and face_cy < cy) or
        (target == "tr" and face_cx >= cx and face_cy < cy) or
        (target == "bl" and face_cx < cx and face_cy >= cy) or
        (target == "br" and face_cx >= cx and face_cy >= cy)
    )
    inside_target = in_center_box if target == "c" else in_quadrant

    aligned = inside_target and dist <= near_unit
    near = inside_target and not aligned and dist <= base_unit

    if aligned:
        return ("Good. Hold steady.", True, False)

    if near:
        return ("Good, almost there.", False, True)

    # Directional guidance with nuance
    horizontal = ""
    vertical = ""
    if abs(dx) > base_unit:
        horizontal = "Move right." if dx > 0 else "Move left."
    elif abs(dx) > near_unit:
        horizontal = "Move slightly to the right." if dx > 0 else "Move slightly to the left."

    if abs(dy) > base_unit:
        vertical = "Move down." if dy > 0 else "Move up."
    elif abs(dy) > near_unit:
        vertical = "Move slightly down." if dy > 0 else "Move slightly up."

    instr = " ".join([p for p in (vertical, horizontal) if p]) or "Please move toward the target section."
    return (instr, False, False)

# ------------------------------
# MAIN LOGIC (Continuous Guidance + Auto Selfie)
# ------------------------------
def main():
    target = get_voice_choice()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Guided Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Guided Camera", 1280, 720)

    last_talk = 0.0
    interval = 2.0  # balanced cadence
    last_instr = None
    aligned_counter = 0  # consecutive aligned confirmations
    phase = 'acquire'  # first ensure user is fully in frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        photo_frame = frame.copy()  # save a clean copy for the selfie

        frame, (x1, y1, x2, y2, cx, cy) = draw_layout(frame, target)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) > 0:
            (fx, fy, fw, fh) = faces[0]
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            face_cx = fx + fw // 2
            face_cy = fy + fh // 2
            current = detect_section(face_cx, face_cy, frame.shape[1], frame.shape[0], x1, y1, x2, y2, cx, cy)

            # Simple frontal check using eyes detection
            roi_gray = gray[fy:fy + fh, fx:fx + fw]
            eyes = []
            try:
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            except Exception:
                eyes = []
            # Consider the face frontal only when both eyes are visible
            frontal_ok = len(eyes) >= 2

            now = time.time()

            # --- Stage 1: Ensure the face is fully inside the frame before target guidance ---
            w_frame, h_frame = frame.shape[1], frame.shape[0]
            safe_margin = int(0.06 * min(w_frame, h_frame))
            fully_inside = (
                fx > safe_margin and fy > safe_margin and
                fx + fw < w_frame - safe_margin and fy + fh < h_frame - safe_margin
            )
            large_enough = (fw >= 0.15 * w_frame) or (fh >= 0.15 * h_frame)

            if phase == 'acquire':
                if now - last_talk > interval:
                    if not fully_inside or not large_enough:
                        nudges = []
                        if fx <= safe_margin:
                            nudges.append("Move right.")
                        if fx + fw >= w_frame - safe_margin:
                            nudges.append("Move left.")
                        if fy <= safe_margin:
                            nudges.append("Move down.")
                        if fy + fh >= h_frame - safe_margin:
                            nudges.append("Move up.")
                        if large_enough is False:
                            nudges.append("Move closer to the camera.")

                        instr = " ".join(nudges) or "Please move the camera until your whole face is in the frame."
                        if instr != last_instr:
                            speak(instr)
                            last_instr = instr
                            last_talk = now
                    else:
                        speak(f"Great, I can see you clearly. Now move to {_section_label(target)}.")
                        phase = 'target'
                        last_instr = None
                        last_talk = now

                # Skip target guidance until acquisition is complete
                continue

            # If we had a face and then lost good framing, fall back to acquire
            if phase == 'target' and (not fully_inside or not large_enough):
                if now - last_talk > interval:
                    speak("You're partially out of view. Re-center yourself, then we'll continue.")
                    last_talk = now
                    last_instr = None
                phase = 'acquire'
                continue

            # --- Stage 2: Target-specific guidance ---
            if now - last_talk > interval:
                instr, is_aligned, _is_near = guidance(
                    face_cx, face_cy, current, target,
                    (w_frame, h_frame),
                    (x1, y1, x2, y2, cx, cy),
                )

                # When inside the target section but face is angled, ask for head rotation
                inside_target = (
                    (target == 'c' and x1 <= face_cx <= x2 and y1 <= face_cy <= y2) or
                    (target == 'tl' and face_cx < cx and face_cy < cy) or
                    (target == 'tr' and face_cx >= cx and face_cy < cy) or
                    (target == 'bl' and face_cx < cx and face_cy >= cy) or
                    (target == 'br' and face_cx >= cx and face_cy >= cy)
                )

                if inside_target and not frontal_ok:
                    rotate_instr = None
                    if len(eyes) == 1:
                        ex, ey, ew, eh = eyes[0]
                        eye_center_x = ex + ew // 2
                        if eye_center_x < fw // 2:
                            rotate_instr = "Rotate your head slightly to your right."
                        else:
                            rotate_instr = "Rotate your head slightly to your left."
                    elif len(eyes) == 0:
                        rotate_instr = "Turn your face toward the camera until both eyes are visible."

                    if rotate_instr:
                        instr = rotate_instr

                # Avoid repeating the exact same instruction too frequently
                if instr != last_instr or is_aligned:
                    speak(instr)
                    last_instr = instr
                    last_talk = now

                # Require both alignment and frontal face before auto-capture
                if is_aligned and frontal_ok:
                    aligned_counter += 1
                    if aligned_counter >= 2:
                        speak("Stop moving. Capturing your selfie.")
                        img_name = f"selfie_{int(time.time())}.jpg"
                        cv2.imwrite(img_name, photo_frame)
                        speak("Selfie captured successfully. Camera closing.")
                        break
                else:
                    aligned_counter = 0
        else:
            # No face detected at all — guide the user to enter the frame first
            now = time.time()
            if now - last_talk > interval:
                speak("I can't see your face. Move the camera slowly left, right, up, and down until I can see you.")
                last_talk = now
                last_instr = None

        cv2.imshow("Guided Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)

if __name__ == "__main__":
    main()
