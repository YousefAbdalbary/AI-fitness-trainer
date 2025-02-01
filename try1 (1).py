import time
import cv2
import threading
import numpy as np
import pyttsx3
import queue
import mediapipe as mp
import speech_recognition as sr

# ------------------------------
# Initialize Mediapipe Pose detection
# ------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------------------
# Constants and Parameters
# ------------------------------
VIDEO_WIDTH, VIDEO_HEIGHT = 1020, 500
ANGLE_UP_THRESH = 150
ANGLE_DOWN_THRESH = 90
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5
FONT_SCALE = 1
FONT_THICKNESS = 2

# Exercise states and counters
state_flags = {
    "push_up_left": False,
    "push_up_right": False,
    "combine": False,
    "pull_up": False,
    "sit_up": False,
    "squat": False,
    "lunge_left": False,
    "lunge_right": False
}

counters = {
    "left_hand": 0,
    "right_hand": 0,
    "combine": 0,
    "pull_up": 0,
    "sit_up": 0,
    "squat": 0,
    "lunge_left": 0,
    "lunge_right": 0
}

mode = None

# ------------------------------
# Initialize Speech Engine
# ------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('voice', engine.getProperty('voices')[1].id)
speech_queue = queue.Queue()

# ------------------------------
# Mediapipe Pose Model Initialization
# ------------------------------
pose = mp_pose.Pose(
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)

# ------------------------------
# Utility Functions
# ------------------------------

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points in 2D space.

    Args:
        a (tuple or list or np.ndarray): Coordinates of the first point (x, y).
        b (tuple or list or np.ndarray): Coordinates of the second point (x, y) (vertex of the angle).
        c (tuple or list or np.ndarray): Coordinates of the third point (x, y).

    Returns:
        float: The angle in degrees between the three points, ranging from 0 to 180.
    
    Raises:
        ValueError: If input points are not valid 2D coordinates.
    """
    # Validate inputs
    for point in [a, b, c]:
        if not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 2:
            raise ValueError(f"Invalid input point: {point}. Each point must be a 2D coordinate (x, y).")

    # Convert points to numpy arrays for efficient computation
    a, b, c = map(np.array, (a, b, c))
    
    # Calculate vectors
    ba = a - b  # Vector from b to a
    bc = c - b  # Vector from b to c

    # Normalize vectors
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)

    if ba_norm == 0 or bc_norm == 0:
        raise ValueError("Zero-length vector encountered. Points must not overlap.")

    ba_unit = ba / ba_norm
    bc_unit = bc / bc_norm

    # Compute the cosine of the angle using the dot product
    cos_theta = np.clip(np.dot(ba_unit, bc_unit), -1.0, 1.0)  # Clipping to handle numerical precision issues

    # Calculate the angle in degrees
    angle = np.degrees(np.arccos(cos_theta))

    return angle


def draw_pose_landmarks(frame, landmarks):
    """Draw landmarks and connections on the frame."""
    if landmarks:
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
    else:
        print("Error: Invalid landmarks detected.")

def speak(text):
    """
    Queue text for speech synthesis.

    Args:
        text (str): The text to be spoken.
    """
    if not text:
        print("No text provided for speech.")
        return
    speech_queue.put(text)

# Worker thread to process speech queue
# Speech synthesis worker thread with graceful shutdown
def worker_speak():
    """
    Process the speech queue and speak text.
    Stops when `None` is added to the queue.
    """
    while True:
        text = speech_queue.get()
        if text is None:  # Stop signal
            print("Speech synthesis worker stopped.")
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
def set_mode(new_mode):
    global mode
    mode = new_mode
    print(f"Mode set to {mode}")
# Function to listen for voice commands
def listen_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = recognizer.listen(source)
                commands = recognizer.recognize_google(audio).lower()
                print(f"Command recognized: {commands}")

                # Voice command to switch to specific exercises
                if 'normal' in commands:
                    speak("Normal mode started")
                    set_mode('normal')
                elif 'start' in commands:
                    speak("Combine mode started")
                    set_mode('combine')
                elif 'squat' in commands:
                    speak("Squat mode started")
                    set_mode('squats')
                elif 'sit-up' in commands:
                    speak("Sit-up mode started")
                    set_mode('sit-ups')
                elif 'lunge' in commands:
                    speak("Lunge mode started")
                    set_mode('lunges')
                elif 'push-up' in commands:
                    speak("Push-up mode started")
                    set_mode('normal')
                elif 'stop' in commands:
                    speak("Take care and goodbye")
                    set_mode('stop')
                    break

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Could not request results; check your network connection")

# Start the worker thread
threading.Thread(target=worker_speak, daemon=True).start()

# Command processing
def process_command(command):
    """
    Process voice commands to set modes or stop the app.
    """
    global mode

    # Define mode mappings for commands
    mode_mappings = {
        "normal": ("normal", "Normal mode activated"),
        "combine": ("combine", "Combine mode activated"),
        "squat": ("squats", "Squat mode activated"),
        "sit-up": ("sit-ups", "Sit-up mode activated"),
        "lunge": ("lunges", "Lunge mode activated"),
        "push-up": ("push_up", "Push-up mode activated"),
        "stop": ("stop", "Stopping application. Goodbye!")
    }

    # Process the command
    for keyword, (new_mode, response) in mode_mappings.items():
        if keyword in command:
            mode = new_mode
            speak(response)
            print(f"Mode updated: {new_mode}")
            
            # Handle special "stop" case
            if new_mode == "stop":
                print("Exiting application...")
                speak("Exiting application. Goodbye!")
                break  # Ensure we exit the loop and stop the app
            
            break
    else:
        # If no command is matched
        speak("Command not recognized. Please try again.")
        print(f"Unrecognized command: {command}")


def update_mode(new_mode, message):
    """
    Update the exercise mode and notify the user.

    Args:
        new_mode (str): The new mode to set.
        message (str): The notification message to convey.
    """
    global mode

    if not isinstance(new_mode, str) or not new_mode.strip():
        print("Error: Invalid mode specified.")
        speak("An error occurred while updating the mode. Please try again.")
        return

    # Update the mode and provide feedback
    previous_mode = mode  # Store the previous mode for debugging/logging
    mode = new_mode
    speak(message)
    print(f"Mode updated from '{previous_mode}' to '{new_mode}' with message: '{message}'")



def speak_advice(message):
    """Function to provide feedback or advice via voice."""
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

# last_feedback_time = {
#     "sit_up": 0,
#     "squat": 0,
#     "push_up": 0,
#     "pull_up" :0,
#     "lunge": 0,
#     "plank": 0,

# }

FEEDBACK_INTERVAL = 3  # Time interval (in seconds) between feedback
last_feedback_time = {}

def check_angle_for_correction(angle, min_angle, max_angle, exercise_name, joint_name):
    """Check if the angle is within the acceptable range and provide feedback if not, at regular intervals."""
    current_time = time.time()  # Get the current time in seconds
    
    # Initialize the time for new exercises if not already in the dictionary
    if exercise_name not in last_feedback_time:
        last_feedback_time[exercise_name] = current_time
    
    # Check if enough time has passed since the last feedback for the specific exercise
    if current_time - last_feedback_time[exercise_name] >= FEEDBACK_INTERVAL:
        if angle < min_angle:
            speak_advice(f"Your {joint_name} angle is too small during {exercise_name}. Try to increase the angle to at least {min_angle} degrees.")
        elif angle > max_angle:
            speak_advice(f"Your {joint_name} angle is too large during {exercise_name}. Try to reduce the angle to below {max_angle} degrees.")
        
        # Update the last feedback time for the specific exercise
        last_feedback_time[exercise_name] = current_time

# ------------------------------
# Exercise Logic
# ------------------------------

def process_push_ups(keypoints):
    """Track push-up counters for left and right arms."""
    global state_flags, counters
    left_angle = calculate_angle(*get_body_angles(keypoints, 5, 7, 9))
    right_angle = calculate_angle(*get_body_angles(keypoints, 6, 8, 10))

    # Define the angle thresholds for push-ups
    ANGLE_UP_THRESH_LEFT = 180
    ANGLE_DOWN_THRESH_LEFT = 70
    ANGLE_UP_THRESH_RIGHT = 180
    ANGLE_DOWN_THRESH_RIGHT = 70

    # Track left arm push-ups
    if left_angle < ANGLE_DOWN_THRESH_LEFT and not state_flags["push_up_left"]:
        state_flags["push_up_left"] = True
    elif left_angle > ANGLE_UP_THRESH_LEFT and state_flags["push_up_left"]:
        counters["left_hand"] += 1
        state_flags["push_up_left"] = False
        speak(f"Left count: {counters['left_hand']}")

    # Track right arm push-ups
    if right_angle < ANGLE_DOWN_THRESH_RIGHT and not state_flags["push_up_right"]:
        state_flags["push_up_right"] = True
    elif right_angle > ANGLE_UP_THRESH_RIGHT and state_flags["push_up_right"]:
        counters["right_hand"] += 1
        state_flags["push_up_right"] = False
        speak(f"Right count: {counters['right_hand']}")

def process_combine_push_ups(keypoints):
    """Track combined push-ups counter."""
    global state_flags, counters
    left_angle = calculate_angle(*get_body_angles(keypoints, 5, 7, 9))
    right_angle = calculate_angle(*get_body_angles(keypoints, 6, 8, 10))

    # Define the angle thresholds for combined push-ups
    ANGLE_UP_THRESH = 180
    ANGLE_DOWN_THRESH = 70

    # Only count combined push-ups when both arms are at the down position
    if left_angle <= ANGLE_DOWN_THRESH and right_angle <= ANGLE_DOWN_THRESH and not state_flags["combine"]:
        state_flags["combine"] = True
    # Count when both arms return to the up position
    elif left_angle >= ANGLE_UP_THRESH and right_angle >= ANGLE_UP_THRESH and state_flags["combine"]:
        counters["combine"] += 1
        state_flags["combine"] = False
        speak(f"Combine count: {counters['combine']}")

def process_lunges(keypoints):
    """Track lunges counter for left and right legs."""
    global state_flags, counters
    left_knee_angle = calculate_angle(*get_body_angles(keypoints, 23, 25, 27))  # Left knee
    right_knee_angle = calculate_angle(*get_body_angles(keypoints, 24, 26, 28))  # Right knee
    check_angle_for_correction(left_knee_angle, 90, 160, "left knee")
    check_angle_for_correction(right_knee_angle, 90, 160, "right knee")

    # Define the angle thresholds for lunges
    ANGLE_UP_THRESH_LEFT = 160
    ANGLE_DOWN_THRESH_LEFT = 90
    ANGLE_UP_THRESH_RIGHT = 160
    ANGLE_DOWN_THRESH_RIGHT = 90

    # Track left leg lunges
    if left_knee_angle < ANGLE_DOWN_THRESH_LEFT and not state_flags["lunge_left"]:
        state_flags["lunge_left"] = True
    elif left_knee_angle > ANGLE_UP_THRESH_LEFT and state_flags["lunge_left"]:
        counters["lunge_left"] += 1
        state_flags["lunge_left"] = False
        speak(f"Left lunge count: {counters['lunge_left']}")

    # Track right leg lunges
    if right_knee_angle < ANGLE_DOWN_THRESH_RIGHT and not state_flags["lunge_right"]:
        state_flags["lunge_right"] = True
    elif right_knee_angle > ANGLE_UP_THRESH_RIGHT and state_flags["lunge_right"]:
        counters["lunge_right"] += 1
        state_flags["lunge_right"] = False
        speak(f"Right lunge count: {counters['lunge_right']}")

def process_pull_ups(keypoints):
    """Track pull-ups counter."""
    global state_flags, counters
    # Define the angle thresholds for pull-ups
    ANGLE_UP_THRESH = 180
    ANGLE_DOWN_THRESH = 45
    elbow_angle = calculate_angle(*get_body_angles(keypoints, 5, 7, 9))  # Left arm
    shoulder_angle = calculate_angle(*get_body_angles(keypoints, 11, 5, 7))  # Shoulder joint

    check_angle_for_correction(elbow_angle, 70, 180, "elbow angle")
    # Feedback on shoulder angle for pull-ups
    check_angle_for_correction(shoulder_angle, 45, 180, "shoulder angle")

    # Count pull-ups when elbow goes below threshold
    if elbow_angle < ANGLE_DOWN_THRESH and not state_flags.get("pull_up"):
        state_flags["pull_up"] = True
    elif elbow_angle > ANGLE_UP_THRESH and state_flags.get("pull_up"):
        counters["pull_up"] = counters.get("pull_up", 0) + 1
        state_flags["pull_up"] = False
        speak(f"Pull-up count: {counters['pull_up']}")

def process_sit_ups(keypoints):
    """Track sit-ups counter and provide feedback based on joint angles."""
    global state_flags, counters
    # Define the angle thresholds for sit-ups
    ANGLE_UP_THRESH = 150
    ANGLE_DOWN_THRESH = 45
    # Calculate hip and torso angles using joint positions
    hip_angle = calculate_angle(*get_body_angles(keypoints, 11, 23, 25))  # Hip joint
    torso_angle = calculate_angle(*get_body_angles(keypoints, 12, 24, 26))  # Torso angle

    # Check if the angles are within acceptable ranges for a proper sit-up
    check_angle_for_correction(hip_angle, 30, 150, "hip angle")  # Example thresholds for hip angle
    check_angle_for_correction(torso_angle, 0, 90, "torso angle")  # Example thresholds for torso angle

    # Track sit-ups by hip angle movement
    if hip_angle < ANGLE_DOWN_THRESH and not state_flags.get("sit_up"):
        state_flags["sit_up"] = True
    elif hip_angle > ANGLE_UP_THRESH and state_flags.get("sit_up"):
        counters["sit_up"] = counters.get("sit_up", 0) + 1
        state_flags["sit_up"] = False
        speak(f"Sit-up count: {counters['sit_up']}")

def process_squats(keypoints):
    """Track squats counter and provide feedback based on joint angles."""
    global state_flags, counters

    # Define the angle thresholds for squats
    ANGLE_UP_THRESH = 150
    ANGLE_DOWN_THRESH = 90

    # Calculate knee and hip angles using joint positions
    knee_angle = calculate_angle(*get_body_angles(keypoints, 23, 25, 27))  # Right knee
    hip_angle = calculate_angle(*get_body_angles(keypoints, 11, 23, 25))  # Right hip

    # Track squats by knee angle movement
    if knee_angle < ANGLE_DOWN_THRESH and not state_flags.get("squat"):
        state_flags["squat"] = True
    elif knee_angle > ANGLE_UP_THRESH and state_flags.get("squat"):
        counters["squat"] = counters.get("squat", 0) + 1
        state_flags["squat"] = False
        speak(f"Squat count: {counters['squat']}")



        
def get_body_angles(keypoints, p1, p2, p3):
    """Retrieve the coordinates for calculating angles."""
    return (
        (keypoints[p1].x, keypoints[p1].y),
        (keypoints[p2].x, keypoints[p2].y),
        (keypoints[p3].x, keypoints[p3].y)
    )

def score_table_on_video(frame, exercise_mode):
    """Display the score table on the video frame for the current exercise."""
    y_offset = 30  # Initial y-offset for text
    
    # Set the font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)  # White color
    thickness = 2

    # Display exercise specific count
    if exercise_mode == "push_ups":
        cv2.putText(frame, f"Push-ups (Left): {counters['left_hand']}", (10, y_offset), font, font_scale, color, thickness)
        y_offset += 30
        cv2.putText(frame, f"Push-ups (Right): {counters['right_hand']}", (10, y_offset), font, font_scale, color, thickness)
    elif exercise_mode == "combine":
        cv2.putText(frame, f"Combined Push-ups: {counters['combine']}", (10, y_offset), font, font_scale, color, thickness)
    elif exercise_mode == "lunges":
        cv2.putText(frame, f"Lunges (Left): {counters['lunge_left']}", (10, y_offset), font, font_scale, color, thickness)
        y_offset += 30
        cv2.putText(frame, f"Lunges (Right): {counters['lunge_right']}", (10, y_offset), font, font_scale, color, thickness)
    elif exercise_mode == "sit_ups":
        cv2.putText(frame, f"Sit-ups: {counters['sit_up']}", (10, y_offset), font, font_scale, color, thickness)
    elif exercise_mode == "squats":
        cv2.putText(frame, f"Squats: {counters['squat']}", (10, y_offset), font, font_scale, color, thickness)
    elif exercise_mode == "pull_ups":
        cv2.putText(frame, f"Pull-ups: {counters['pull_up']}", (10, y_offset), font, font_scale, color, thickness)
    else:
        cv2.putText(frame, "No exercise selected.", (10, y_offset), font, font_scale, color, thickness)

# ------------------------------
# Main Loop
# ------------------------------

def main():
    global mode
    cap = cv2.VideoCapture(0)

    # Start threads for speech synthesis and listening
    threading.Thread(target=worker_speak, daemon=True).start()
    threading.Thread(target=listen_commands, daemon=True).start()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            draw_pose_landmarks(frame, results.pose_landmarks)
            keypoints = results.pose_landmarks.landmark

            # Check mode and process corresponding exercise
            if mode == "normal":
                process_push_ups(keypoints)
            elif mode == "squats":
                process_squats(keypoints)
            elif mode == "lunges":
                process_lunges(keypoints)
            elif mode == "pull_ups":
                process_pull_ups(keypoints)
            elif mode == "sit_ups":
                process_sit_ups(keypoints)
            elif mode == "combine":
                process_combine_push_ups(keypoints)
            elif mode == "stop":
                print("Exercise tracking stopped.")
                break
        
        score_table_on_video(frame, mode)
        # Show the frame with landmarks
        cv2.imshow("Exercise Tracker", frame)

        # Exit condition on 'esc' key or 'stop' mode
        if cv2.waitKey(1) & 0xFF == 27 or mode == "stop":
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#     ANGLE_DOWN_THRESH = 30  # or whatever value suits your needs
# ANGLE_UP_THRESH = 170   # adjust accordingly