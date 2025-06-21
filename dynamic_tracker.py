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

# Global state for exercise tracking
exercise_state = {
    'counter': 0,
    'stage': 'down',
    'running': False,
    'config': None,
    'websocket': None,
    'start_time': None
}

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
        config = json.loads(match.group())
        print("\n[Gemini JSON Output]")
        print(json.dumps(config, indent=4))  # Nicely formatted
        return config
    except Exception as e:
        print(f"[ERROR] Gemini parsing failed: {e}")
        return None

def get_feedback(exercise, reps, duration):
    prompt = f"Stroke rehabilitation. I did {exercise} exercise: {reps} reps in {duration:.2f} seconds. Provide short feedback."
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def get_frames():
    """Generator function to yield camera frames with pose detection"""
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while exercise_state['running'] and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks and exercise_state['config']:
                    landmarks = results.pose_landmarks.landmark
                    config = exercise_state['config']
                    
                    # Get points from config
                    point_a = config["point_a"]
                    point_b = config["point_b"]
                    point_c = config["point_c"]
                    
                    # Check visibility
                    if (landmarks[getattr(mp_pose.PoseLandmark, point_a).value].visibility > 0.5 and 
                        landmarks[getattr(mp_pose.PoseLandmark, point_b).value].visibility > 0.5 and 
                        landmarks[getattr(mp_pose.PoseLandmark, point_c).value].visibility > 0.5):
                        
                        # Get coordinates
                        a = [landmarks[getattr(mp_pose.PoseLandmark, point_a).value].x,
                             landmarks[getattr(mp_pose.PoseLandmark, point_a).value].y]
                        b = [landmarks[getattr(mp_pose.PoseLandmark, point_b).value].x,
                             landmarks[getattr(mp_pose.PoseLandmark, point_b).value].y]
                        c = [landmarks[getattr(mp_pose.PoseLandmark, point_c).value].x,
                             landmarks[getattr(mp_pose.PoseLandmark, point_c).value].y]

                        angle = calculate_angle(a, b, c)
                        
                        # Draw angle on frame
                        cv2.putText(image, str(int(angle)),
                                  tuple(np.multiply(b, [640, 480]).astype(int)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Rep logic
                        contracted_angle = config['contracted_angle']
                        relaxed_angle = config['relaxed_angle']
                        contracted_angle, relaxed_angle = max(contracted_angle, relaxed_angle), min(contracted_angle, relaxed_angle)
                        
                        if angle > contracted_angle and exercise_state['stage'] == "down":
                            exercise_state['stage'] = "up"
                            if exercise_state['counter'] == 0:
                                exercise_state['start_time'] = time.time()
                            exercise_state['counter'] += 1
                            if exercise_state['websocket']:
                                exercise_state['websocket'].send(json.dumps({
                                    'counter': exercise_state['counter'],
                                    'stage': exercise_state['stage']
                                }))

                        elif angle < relaxed_angle and exercise_state['stage'] == "up":
                            exercise_state['stage'] = "down"
                            if exercise_state['websocket']:
                                exercise_state['websocket'].send(json.dumps({
                                    'counter': exercise_state['counter'],
                                    'stage': exercise_state['stage']
                                }))

            except Exception as e:
                print(f"Error processing landmarks: {e}")

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def start_exercise(exercise):
    """Initialize exercise tracking
    Returns: True if successful to connect with gemini, False otherwise"""

    exercise_state['config'] = get_exercise_config(exercise)
    if not exercise_state['config']:
        return False
    
    exercise_state['running'] = True
    exercise_state['counter'] = 0
    exercise_state['stage'] = 'down'
    exercise_state['start_time'] = None
    speak_text(f"Starting {exercise} exercise")
    return True
    
def stop_exercise():
    """Stop exercise tracking and return results"""
    exercise_state['running'] = False
    end_time = time.time()
    duration = end_time - (exercise_state['start_time'] or end_time)
    
    feedback = get_feedback(
        exercise_state['config']['exercise'],
        exercise_state['counter'],
        duration
    )
    speak_text(feedback)
    return {
        'count': exercise_state['counter'],
        'duration': duration,
        'feedback': feedback
    }

def set_websocket(ws):# look into why this is needed and what it does
    """Set websocket for real-time updates"""
    exercise_state['websocket'] = ws