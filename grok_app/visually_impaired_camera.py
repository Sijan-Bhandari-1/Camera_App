import cv2
import pyttsx3
import speech_recognition as sr
import time
import threading
import queue
import os

# ------------------------------
# SPEECH ENGINE
# ------------------------------
speech_queue = queue.Queue()
engine = pyttsx3.init(driverName='sapi5')
voices = engine.getProperty('voices')
for v in voices:
    if 'male' in v.name.lower():
        engine.setProperty('voice', v.id)
        break

engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

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

def listen():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        speak("Listening for your choice...")
        print("[DEBUG] Listening for choice...")
        try:
            audio = recognizer.listen(source, phrase_time_limit=5)
        except Exception as e:
            print("[DEBUG] Audio capture failed:", e)
            return ""
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"[DEBUG] Heard: {text}")
        return text
    except:
        speak("Sorry, I could not understand.")
        return ""

def get_voice_choice():
    speak("Please say your target section: top left, top right, bottom left, bottom right, or center.")
    for attempt in range(3):
        response = listen()
        if "top left" in response:
            speak("You selected top left.")
            return "tl"
        elif "top right" in response:
            speak("You selected top right.")
            return "tr"
        elif "bottom left" in response:
            speak("You selected bottom left.")
            return "bl"
        elif "bottom right" in response:
            speak("You selected bottom right.")
            return "br"
        elif "center" in response:
            speak("You selected center.")
            return "c"
        else:
            speak("I did not catch that. Please try again.")
    speak("Defaulting to center.")
    return "c"

# ------------------------------
# LAYOUT + DETECTION
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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

def guidance(current, target):
    if current == target:
        return "Perfect! Face aligned."
    moves = []
    if current in ["tl", "bl"] and target in ["tr", "br", "c"]:
        moves.append("move right")
    elif current in ["tr", "br"] and target in ["tl", "bl", "c"]:
        moves.append("move left")
    if current in ["tl", "tr"] and target in ["bl", "br", "c"]:
        moves.append("move down")
    elif current in ["bl", "br"] and target in ["tl", "tr", "c"]:
        moves.append("move up")
    return " and ".join(moves)

# ------------------------------
# MAIN LOGIC (Continuous Guidance + Auto Selfie)
# ------------------------------
def main():
    target = get_voice_choice()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Guided Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Guided Camera", 1280, 720)

    last_talk = 0
    interval = 2.5
    aligned_counter = 0  # count consecutive perfect frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        frame, (x1, y1, x2, y2, cx, cy) = draw_layout(frame, target)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) > 0:
            (fx, fy, fw, fh) = faces[0]
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            face_cx = fx + fw // 2
            face_cy = fy + fh // 2
            current = detect_section(face_cx, face_cy, frame.shape[1], frame.shape[0], x1, y1, x2, y2, cx, cy)

            now = time.time()
            if now - last_talk > interval:
                instr = guidance(current, target)
                speak(instr)
                last_talk = now

                if instr == "Perfect! Face aligned.":
                    aligned_counter += 1
                    if aligned_counter >= 2:
                        speak("Do not move now. Capturing your selfie.")
                        img_name = f"selfie_{int(time.time())}.jpg"
                        cv2.imwrite(img_name, frame)
                        speak("Selfie captured successfully. Camera closing.")
                        break
                else:
                    aligned_counter = 0

        cv2.imshow("Guided Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)

if __name__ == "__main__":
    main()
