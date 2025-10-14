import cv2
import pyttsx3
import speech_recognition as sr
import time
import re
import os
import sys
import mediapipe as mp
import tkinter as tk
import threading
# ------------------------------
# CONFIG
# ------------------------------
INSTRUCTION_COOLDOWN = 3.5   # seconds between spoken instructions
MIN_STABLE_ROUNDS = 3        # consecutive aligned frames before capture
WINDOW_SCALE = 0.85          # fraction of screen for window

# Power/Performance tuning
TARGET_FPS = 15              # cap end-to-end processing rate
CAPTURE_WIDTH = 640          # camera capture width
CAPTURE_HEIGHT = 480         # camera capture height
PROCESS_MAX_WIDTH = 400      # downscale width for FaceMesh processing

# Distance thresholds
DIST_NEAR = 0.10
DIST_FAR = 0.04

# ------------------------------
# TTS setup (cross-platform with fallback)
# ------------------------------
try:
    if sys.platform.startswith("win"):
        engine = pyttsx3.init("sapi5")
    elif sys.platform == "darwin":
        engine = pyttsx3.init("nsss")
    else:
        engine = pyttsx3.init()  # espeak on Linux
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    # Prefer familiar voices on Windows, if available
    if sys.platform.startswith("win"):
        voices = engine.getProperty('voices')
        for v in voices:
            if "Zira" in v.name or "David" in v.name:
                engine.setProperty('voice', v.id)
                break
except Exception:
    engine = None
    
def speak(text: str):
    """Speak text immediately (main thread)."""
    if not text:
        return
    print(f"[INSTRUCTION]: {text}")
    if engine is None:
        return
    engine.say(text)
    engine.runAndWait()

# ------------------------------
# Speech recognition for target selection
# ------------------------------
recognizer = sr.Recognizer()
try:
    mic = sr.Microphone()
except Exception:
    mic = None

def listen(phrase_time_limit: int = 5) -> str:
    if mic is None:
        return ""
    with mic as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            audio = recognizer.listen(source, phrase_time_limit=phrase_time_limit, timeout=8)
        except Exception:
            return ""
    try:
        return recognizer.recognize_google(audio).lower()
    except Exception:
        return ""

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_voice_choice() -> str:
    speak("Welcome. Please select your target section: Top Left, Top Right, Bottom Left, Bottom Right, or Center.")
    time.sleep(1.5)
    
    for attempt in range(3):
        print(f"[LISTENING] Attempt {attempt+1}/3...")
        response = normalize(listen())
        if not response:
            speak("I did not catch that. Please say again.")
            time.sleep(1.5)
            continue
        print(f"[HEARD]: {response}")
        if "top left" in response or "upper left" in response:
            speak("Target section selected: Top Left.")
            return "tl"
        if "top right" in response or "upper right" in response:
            speak("Target section selected: Top Right.")
            return "tr"
        if "bottom left" in response or "lower left" in response:
            speak("Target section selected: Bottom Left.")
            return "bl"
        if "bottom right" in response or "lower right" in response:
            speak("Target section selected: Bottom Right.")
            return "br"
        if "center" in response or "middle" in response:
            speak("Target section selected: Center.")
            return "c"
        speak("I did not understand. Please say again.")
        time.sleep(1.5)
    speak("No selection detected. Defaulting to Center.")
    return "c"

# ------------------------------
# Grid drawing
# ------------------------------
def draw_full_grid(frame, selected):
    h, w, _ = frame.shape
    cx, cy = w//2, h//2
    box_w, box_h = w//3, h//3
    x1, y1 = cx - box_w//2, cy - box_h//2
    x2, y2 = cx + box_w//2, cy + box_h//2

    color = (0,255,0)
    cv2.line(frame, (cx,0),(cx,h), color, 1)
    cv2.line(frame, (0,cy),(w,cy), color, 1)
    cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,255),1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "TL", (10,30), font,0.7,color,2)
    cv2.putText(frame, "TR", (w-50,30), font,0.7,color,2)
    cv2.putText(frame, "BL", (10,h-20), font,0.7,color,2)
    cv2.putText(frame, "BR", (w-50,h-20), font,0.7,color,2)

    overlay = frame.copy()
    alpha = 0.25
    highlight = (0,255,255)
    if selected=="tl":
        cv2.rectangle(overlay, (0,0),(cx,cy),highlight,-1)
    elif selected=="tr":
        cv2.rectangle(overlay, (cx,0),(w,cy),highlight,-1)
    elif selected=="bl":
        cv2.rectangle(overlay, (0,cy),(cx,h),highlight,-1)
    elif selected=="br":
        cv2.rectangle(overlay, (cx,cy),(w,h),highlight,-1)
    elif selected=="c":
        cv2.rectangle(overlay, (x1,y1),(x2,y2),highlight,-1)

    return cv2.addWeighted(overlay,alpha,frame,1-alpha,0), (x1,y1,x2,y2,cx,cy)

def _target_center(selected, w,h,cx,cy,x1,y1,x2,y2):
    if selected=="tl": return (cx//2, cy//2)
    if selected=="tr": return (cx+(w-cx)//2, cy//2)
    if selected=="bl": return (cx//2, cy+(h-cy)//2)
    if selected=="br": return (cx+(w-cx)//2, cy+(h-cy)//2)
    return ((x1+x2)//2, (y1+y2)//2)

# ------------------------------
# Head pose instructions
# ------------------------------
mp_face = mp.solutions.face_mesh

def get_head_pose_instructions(landmarks, face_bbox, w,h):
    lx, ly = int(landmarks[33].x*w), int(landmarks[33].y*h)
    rx, ry = int(landmarks[263].x*w), int(landmarks[263].y*h)
    nx, ny = int(landmarks[1].x*w), int(landmarks[1].y*h)
    cx, cy, cw, ch = face_bbox
    center_x, center_y = cx + cw//2, cy + ch//2

    instr = []
    dx = nx - center_x
    if abs(dx) > 0.05*cw: instr.append("Rotate right." if dx<0 else "Rotate left.")
    dy = ny - center_y
    if abs(dy) > 0.05*ch: instr.append("Tilt down." if dy<0 else "Tilt up.")
    eye_dx = rx-lx
    eye_dy = ry-ly
    if eye_dx!=0 and abs(eye_dy/eye_dx)>0.12: instr.append("Straighten your head.")
    return " ".join(instr) or "Face looks good."

# ------------------------------
# Main loop
# ------------------------------
def main():
    # Reduce OpenCV CPU usage
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)
    except Exception:
        pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not available.")
        return
    # Apply capture constraints to lower sensor and decode power
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    os.makedirs("selfies", exist_ok=True)

    # Screen setup
    root = tk.Tk()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    win_w, win_h = int(screen_w*WINDOW_SCALE), int(screen_h*WINDOW_SCALE)
    win_x, win_y = (screen_w-win_w)//2, (screen_h-win_h)//2
    cv2.namedWindow("Guided Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Guided Camera", win_w, win_h)
    try: cv2.moveWindow("Guided Camera", win_x, win_y)
    except: pass

    # Target selection
    target = get_voice_choice()
    speak("Starting camera guidance. Follow my instructions.")

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        last_instruction_time = 0.0
        aligned_history = []
        min_frame_interval = 1.0 / max(1, TARGET_FPS)

        while True:
            iter_start = time.time()
            ret, frame = cap.read()
            if not ret:
                speak("Lost camera feed.")
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame, layout_rect = draw_full_grid(frame, target)
            x1,y1,x2,y2,cx,cy = layout_rect

            # Downscale before heavy processing to reduce CPU usage
            process_bgr = frame
            if w > PROCESS_MAX_WIDTH:
                scale = PROCESS_MAX_WIDTH / float(w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                process_bgr = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            process_rgb = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(process_rgb)

            now = time.time()
            can_generate = (now - last_instruction_time) >= INSTRUCTION_COOLDOWN
            instruction_parts = []

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                xs, ys = [int(l.x*w) for l in landmarks], [int(l.y*h) for l in landmarks]
                fx, fy = min(xs), min(ys)
                fw, fh = max(xs)-min(xs), max(ys)-min(ys)
                face_bbox = (fx, fy, fw, fh)
                face_cx, face_cy = fx+fw//2, fy+fh//2

                cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)

                tx, ty = _target_center(target,w,h,cx,cy,x1,y1,x2,y2)
                inside_target = (fx>=x1 and fy>=y1 and fx+fw<=x2 and fy+fh<=y2)
                head_instr = get_head_pose_instructions(landmarks, face_bbox, w,h)
                aligned_head = (head_instr=="Face looks good.")

                # Movement
                dx, dy = tx-face_cx, ty-face_cy
                base_unit, near_unit = 0.08*min(w,h), 0.04*min(w,h)
                if not inside_target:
                    if abs(dx)>base_unit: instruction_parts.append("Move right." if dx>0 else "Move left.")
                    elif abs(dx)>near_unit: instruction_parts.append("Move slightly right." if dx>0 else "Move slightly left.")
                    if abs(dy)>base_unit: instruction_parts.append("Move down." if dy>0 else "Move up.")
                    elif abs(dy)>near_unit: instruction_parts.append("Move slightly down." if dy>0 else "Move slightly up.")

                # Distance
                face_ratio = (fw*fh)/(w*h)
                if face_ratio>DIST_NEAR: instruction_parts.append("Move farther from the camera.")
                elif face_ratio<DIST_FAR: instruction_parts.append("Move closer to the camera.")

                # Head pose
                if inside_target and head_instr!="Face looks good.": instruction_parts.append(head_instr)

                # Speak instructions
                if can_generate and instruction_parts:
                    instruction_text = " ".join(instruction_parts)
                    speak(instruction_text)
                    last_instruction_time = now

                # Alignment
                aligned_history.append(inside_target and aligned_head)
                if len(aligned_history)>6: aligned_history.pop(0)

                # Capture selfie
                if len(aligned_history)>=MIN_STABLE_ROUNDS and all(aligned_history[-MIN_STABLE_ROUNDS:]):
                    speak("Perfect! Hold still. Capturing in three, two, one.")
                    time.sleep(2.5)
                    img_name = f"selfies/selfie_{int(time.time())}.jpg"
                    # Capture a fresh, clean frame for saving (no overlays)
                    ret_save, save_frame = cap.read()
                    if ret_save:
                        save_frame = cv2.flip(save_frame, 1)
                        selfie_crop = save_frame[y1:y2, x1:x2]
                    else:
                        # Fallback to current frame region (may contain overlays)
                        selfie_crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(img_name, selfie_crop)
                    speak("Selfie captured successfully. Camera closing.")
                    time.sleep(2.0)
                    break

            else:
                if can_generate:
                    speak("I cannot see your face. Please move into the frame.")
                    last_instruction_time = now

            cv2.imshow("Guided Camera", frame)
            # Throttle to target FPS using waitKey delay
            elapsed = time.time() - iter_start
            remaining = max(0.0, min_frame_interval - elapsed)
            wait_ms = max(1, int(remaining * 1000))
            if cv2.waitKey(wait_ms) & 0xFF==ord("q"):
                speak("Application closed by user.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
