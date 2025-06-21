import cv2
import time
import json
import re
import numpy as np
import pyttsx3
import threading
import mediapipe as mp
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyCgHnk-X0PQPvqZakHLkb0R4ZMCsB5k5tA")

# Initialize MediaPipe and Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def get_exercise_config(exercise):
    prompt = f"""For the exercise {exercise}, give me the three MediaPipe pose landmarks (point A, point B, and point C) to calculate an angle for rep counting.

- The landmarks should be ordered from top to bottom (e.g., shoulder → elbow → wrist).
- Also provide the approximate angle values in degrees for:
  - the "up" position (rep at the top)
  - the "down" position (rep at the bottom)

Use MediaPipe's official landmark names from the mp_pose.PoseLandmark enum.

Output the result in this flat JSON format:
{{
  "exercise": "{exercise}",
  "point_a": "LANDMARK_NAME",
  "point_b": "LANDMARK_NAME",
  "point_c": "LANDMARK_NAME",
  "up_angle_threshold": VALUE,
  "down_angle_threshold": VALUE
}}"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    try:
        match = re.search(r'{.*}', response.text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response.")
        config = json.loads(match.group())
        print("\n[Gemini JSON Output]")
        print(json.dumps(config, indent=4))  # Nicely formatted
        return config
    except Exception as e:
        print(f"[ERROR] Gemini parsing failed: {e}")
        return None

def get_feedback(exercise, reps, duration):
    prompt = f"Stroke rehabilitation. I did {exercise} exercise: {reps} reps in {duration:.2f} seconds. Provide short feedback."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def run_exercise(exercise):
    config = get_exercise_config(exercise)
    if not config:
        return 0, 0, "Could not get configuration from Gemini."

    # Unpack config
    point_a = config["point_a"]
    point_b = config["point_b"]
    point_c = config["point_c"]
    up_thresh = config["up_angle_threshold"]
    down_thresh = config["down_angle_threshold"]

    print(f"[CONFIG] A={point_a}, B={point_b}, C={point_c}, Up={up_thresh}, Down={down_thresh}")
    speak_text(f"Starting {exercise} exercise")

    cap = cv2.VideoCapture(0)
    counter = 0
    stage = "down"
    start_time = None
    end_time = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                a = [landmarks[getattr(mp_pose.PoseLandmark, point_a).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, point_a).value].y]
                b = [landmarks[getattr(mp_pose.PoseLandmark, point_b).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, point_b).value].y]
                c = [landmarks[getattr(mp_pose.PoseLandmark, point_c).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, point_c).value].y]
                left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
                right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value].y

                angle = calculate_angle(a, b, c)
                cv2.putText(image, str(int(angle)),
                            tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Rep logic
                if angle > up_thresh and stage == "down" and left_heel < 1 and right_heel < 1 and nose > 0:
                    stage = "up"
                    if counter == 0:
                        start_time = time.time()
                    counter += 1

                elif angle < down_thresh and stage == "up" and left_heel < 1 and right_heel < 1 and nose > 0:
                    stage = "down"
                    threading.Thread(target=speak_text, args=(str(counter),)).start()

            except Exception as e:
                print(f"[Pose Error] {e}")

            # Display counter
            cv2.rectangle(image, (0, 0), (250, 70), (245, 117, 16), -1)
            cv2.putText(image, f"{counter} {stage}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Exercise Tracker', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if start_time:
        end_time = time.time()
        duration = end_time - start_time
    else:
        duration = 0

    feedback = get_feedback(exercise, counter, duration)
    threading.Thread(target=speak_text, args=(feedback,)).start()

    return counter, duration, feedback
