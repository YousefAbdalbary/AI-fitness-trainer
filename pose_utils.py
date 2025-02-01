import math
import mediapipe as mp

mp_pose = mp.solutions.pose

def get_body_length(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    left_distance = calculate_distance(nose, left_hip)
    right_distance = calculate_distance(nose, right_hip)

    return (left_distance + right_distance) / 2

def calculate_distance(landmark1, landmark2):
    x1, y1, z1 = landmark1.x, landmark1.y, landmark1.z
    x2, y2, z2 = landmark2.x, landmark2.y, landmark2.z
    return math.sqrt((x2 - x1) * 2 + (y2 - y1) * 2 + (z2 - z1) ** 2)

def calculate_healthy_weight(height, weight=None):
    height_m = height / 100
    bmi_lower = 18.5 * (height_m ** 2)
    bmi_upper = 24.9 * (height_m ** 2)
    
    if weight:
        bmi = weight / (height_m ** 2)
        status = 'underweight' if bmi < 18.5 else 'healthy' if bmi <= 24.9 else 'overweight'
        return (bmi_lower, bmi_upper, bmi, status)
    
    return (bmi_lower, bmi_upper)