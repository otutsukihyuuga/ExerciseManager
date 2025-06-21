import cv2
import time
import json
import re
import numpy as np
import pyttsx3
import threading
import mediapipe as mp
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

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
    prompt = f"""For the exercise {exercise}, give me the three MediaPipe pose landmarks (point A, point B, and point C) to calculate angle for rep counting.

- The landmarks should be ordered from top to bottom (e.g., shoulder → elbow → wrist).
- Also provide the approximate angle values in degrees for:
  - the contracted muscle angle
  - the relaxed muscle angle

Use MediaPipe's official landmark names from the mp_pose.PoseLandmark enum.

Output the result in this flat JSON format:
{{
  "exercise": "{exercise}",
  "point_a": "LANDMARK_NAME",
  "point_b": "LANDMARK_NAME",
  "point_c": "LANDMARK_NAME",
  "contracted_angle": VALUE,
  "relaxed_angle": VALUE
}}"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    try:
        match = re.search(r'{.*}', response.text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in response.")
        return json.loads(match.group())
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
    else:
        print(f"[GEMINI] Exercise: {exercise}")
        print(f"[GEMINI] Config: {config}")

    # Unpack config
    point_a = config["point_a"]
    point_b = config["point_b"]
    point_c = config["point_c"]
    contracted_angle = config["contracted_angle"]
    relaxed_angle = config["relaxed_angle"]

    print(f"[CONFIG] A={point_a}, B={point_b}, C={point_c}, Contracted={contracted_angle}, Relaxed={relaxed_angle}")
    speak_text(f"Starting {exercise} exercise")

    cap = cv2.VideoCapture(0)
    counter = 0
    stage = "down"
    visibility = False
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
                if landmarks[getattr(mp_pose.PoseLandmark, point_a).value].visibility > 0.5 and landmarks[getattr(mp_pose.PoseLandmark, point_b).value].visibility > 0.5 and landmarks[getattr(mp_pose.PoseLandmark, point_c).value].visibility > 0.5:
                    visibility = True
                else:
                    visibility = False 

                a = [landmarks[getattr(mp_pose.PoseLandmark, point_a).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, point_a).value].y]
                b = [landmarks[getattr(mp_pose.PoseLandmark, point_b).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, point_b).value].y]
                c = [landmarks[getattr(mp_pose.PoseLandmark, point_c).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, point_c).value].y]

                angle = calculate_angle(a, b, c)
                cv2.putText(image, str(int(angle)),
                            tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # Rep logic
                contracted_angle, relaxed_angle = max(contracted_angle, relaxed_angle), min(contracted_angle, relaxed_angle)
                if angle > contracted_angle and stage == "down" and visibility:
                    stage = "up"
                    if counter == 0:
                        start_time = time.time()
                    counter += 1

                elif angle < relaxed_angle and stage == "up" and visibility:
                    stage = "down"
                    # threading.Thread(target=speak_text, args=(str(counter),)).start()

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
