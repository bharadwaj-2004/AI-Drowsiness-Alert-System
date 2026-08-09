import cv2
import mediapipe as mp
import math
import time
import pygame

# ---------------- MediaPipe setup ----------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21
CLOSED_TRIGGER_SEC = 4     # eyes must stay closed this long to start siren
OPEN_STOP_SEC = 3          # eyes must stay open this long to stop siren

# ---------------- siren setup (custom MP3 via pygame) ----------------
pygame.mixer.init()
SIREN_FILE = r"C:\Users\Bharadwaj\OneDrive\Desktop\ribhavagrawal-alarm-siren-sound-effect-type-01-294194.mp3"

def start_siren():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(SIREN_FILE)
        pygame.mixer.music.play(loops=-1)  # loop forever until stopped

def stop_siren():
    pygame.mixer.music.stop()
# -----------------------------------------------------------------------


def euclidean(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def eye_aspect_ratio(landmarks, eye_points):
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_points)
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)


eyes_closed_since = None
eyes_open_since = None
alarm_on = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status_text = "No face detected"
    color = (200, 200, 200)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        eyes_closed = avg_ear < EAR_THRESHOLD
        now = time.time()

        if eyes_closed:
            eyes_open_since = None
            if eyes_closed_since is None:
                eyes_closed_since = now

            closed_duration = now - eyes_closed_since
            status_text = f"EYES CLOSED ({closed_duration:.1f}s)"
            color = (0, 0, 255)

            if closed_duration >= CLOSED_TRIGGER_SEC and not alarm_on:
                start_siren()
                alarm_on = True

        else:
            eyes_closed_since = None
            if eyes_open_since is None:
                eyes_open_since = now

            open_duration = now - eyes_open_since
            status_text = "EYES OPEN"
            color = (0, 200, 0)

            if alarm_on:
                remaining = max(0, OPEN_STOP_SEC - open_duration)
                status_text = f"EYES OPEN - stopping siren in {remaining:.1f}s"
                if open_duration >= OPEN_STOP_SEC:
                    stop_siren()
                    alarm_on = False

    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.imshow("Eye State Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_siren()
        break

cap.release()
cv2.destroyAllWindows()