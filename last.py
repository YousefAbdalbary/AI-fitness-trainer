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
        self.counter = 0  # Total correct steps
        self.incorrect_steps = 0  # Total incorrect steps
        self.status = True  # Current status of the exercise (correct/incorrect)
        self.avg_score = 0  # Average score of the exercise
        self.prev_status = False  # Track the previous status to detect transitions

    def set_exercise(self, exercise_type):
        self.exercise_type = exercise_type
        self.counter = 0
        self.incorrect_steps = 0
        self.status = True
        self.avg_score = 0
        self.prev_status = False  # Reset previous status when exercise type changes

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

                # Check for a complete step
                if self.status != self.prev_status:  # Transition detected
                    if self.status:  # Correct step completed
                        self.counter += 1  # Increment counter only on correct step completion
                    else:  # Incorrect step detected
                        self.incorrect_steps += 1  # Increment incorrect steps
                    self.prev_status = self.status  # Update previous status

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

            # Calculate accuracy
            total_steps = self.counter + self.incorrect_steps
            accuracy = (self.counter / total_steps * 100) if total_steps > 0 else 0

            # Display exercise metrics
            cv2.putText(frame, f'Reps: {self.counter}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 6)
            cv2.putText(frame, f'Incorrect Steps: {self.incorrect_steps}', (20, 130), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 6)
            cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (20, 190), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 6)
            cv2.putText(frame, f'Score: {int(self.avg_score)}', (20, 250), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 6)

            # Display feedback
            feedback = "Correct! Keep going." if self.status else "Incorrect form. Adjust your posture."
            cv2.putText(frame, f'Feedback: {feedback}', (20, 310), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        return frame

    def get_stats(self):
        # Calculate accuracy
        total_steps = self.counter + self.incorrect_steps
        accuracy = (self.counter / total_steps * 100) if total_steps > 0 else 0

        return {
            'counter': self.counter,  # Correct steps
            'incorrect_steps': self.incorrect_steps,  # Incorrect steps
            'accuracy': accuracy,  # Accuracy percentage
            'score': int(self.avg_score)  # Average score
        }


def main():
    # Initialize the ExerciseTracker
    tracker = ExerciseTracker()

    # Ask user to choose exercise type
    print("Choose an exercise to track:")
    print("1. Pull-up")
    print("2. Sit-up")
    print("3. Push-up")
    print("4. Squat")
    print("5. Walk")

    exercise_choice = input("Enter the number of the exercise (1-5): ")

    # Map the user choice to the exercise type
    exercise_types = {
        '1': 'pull-up',
        '2': 'sit-up',
        '3': 'push-up',
        '4': 'squat',
        '5': 'walk'
    }
    exercise_type = exercise_types.get(exercise_choice, 'sit-up')  # Default to 'sit-up' if invalid choice

    # Set the exercise type
    tracker.set_exercise(exercise_type)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame = tracker.process_frame(frame)

        # Display the frame
        cv2.imshow('Exercise Tracker', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Get final stats
    stats = tracker.get_stats()
    print("\n--- Exercise Results ---")
    print(f"Correct Steps: {stats['counter']}")
    print(f"Incorrect Steps: {stats['incorrect_steps']}")
    print(f"Accuracy: {stats['accuracy']:.2f}%")
    print(f"Score: {stats['score']}")
    print("------------------------")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()