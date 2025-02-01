import cv2
import mediapipe as mp
from pose_utils import *
from exercise_types import TypeOfExercise

class ExerciseTracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
        self.exercise_type = None
        self.counter = 0
        self.status = True
        self.avg_score = 0

    def set_exercise(self, exercise_type):
        self.exercise_type = exercise_type
        self.counter = 0
        self.status = True
        self.avg_score = 0

    def is_tracking(self):
        return self.exercise_type is not None

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if self.exercise_type:
                self.counter, self.status, self.avg_score = TypeOfExercise(
                    landmarks).calculate_exercise(
                        self.exercise_type, 
                        self.counter, 
                        self.status, 
                        self.avg_score
                    )

            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        return frame

    def get_stats(self):
        return {
            'counter': self.counter,
            'score': int(self.avg_score)
        }
