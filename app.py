from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
from exercise_tracker import ExerciseTracker
from pose_utils import calculate_healthy_weight
import subprocess
import g4f
g4f.debug.logging = False # enable logging
g4f.check_version = False # Disable automatic version checking
app = Flask(__name__)
@app.route('/sendmessage', methods=['POST'])
def sendmessage():
    data = request.get_json()
    messages = [
    {"role": "system", "content": "You are a cutting-edge, AI-powered GYM Assistant, dedicated to helping users achieve peak physical fitness. Your expertise spans personalized, science-backed workout programs, high-performance nutrition plans, and targeted strategies for body sculpting and muscle growth. You provide real-time feedback on workouts, track progress, and recommend optimal health foods to fuel performance and recovery. Your advice is tailored, data-driven, and always focused on maximizing efficiency in training, muscle development, and overall health. You do not engage in any conversations unrelated to fitness, health, or nutrition. Your mission is to help users reach their ultimate fitness potential with precision and dedication.Your responses are structured with easy-to-follow formats that is html only."},
    {"role": "user", "content": data['message']}
    ]
    response = g4f.ChatCompletion.create(
    model=g4f.models.gpt_4,
    messages=messages
    )
    print(response)
    return jsonify({"response":response.replace("```html", "").replace("```", "")})


# @app.route('/run_command', methods=['POST'])
# def run_command():
#     if request.method == 'POST':
#         # The command you want to run, for example, 'ls'
#         command = "python3 try1.py"
#         try:
#             result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             output = result.stdout.decode('utf-8')
#         except subprocess.CalledProcessError as e:
#             output = e.stderr.decode('utf-8')
#         return render_template('index.html', output=output)

@app.route('/run_voice_script', methods=['GET'])
def run_voice_script():
    try:
        # Ensure the Python script is accessible and run it
        subprocess.run(['python3', r'D:\download\download\project\project\try1.py'], check=True)
        return jsonify(success=True)
    except Exception as e:
        print(f"Error running script: {e}")
        return jsonify(success=False)
@app.route('/run_therapy_script', methods=['GET'])
def run_therapy_script():
    try:
        # Run the script using subprocess
        subprocess.run(['python3', r'D:\download\download\project\project\Physio_CV.py'], check=True)
        return jsonify(success=True)
    except Exception as e:
        print(f"Error running the script: {e}")
        return jsonify(success=False)
# Initialize SocketIO without specifying async_mode
socketio = SocketIO(app)

exercise_tracker = ExerciseTracker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_exercise/<exercise_type>')
def start_exercise(exercise_type):
    exercise_tracker.set_exercise(exercise_type)
    return render_template('exercise.html', exercise_type=exercise_type)

@app.route('/measure_body')
def measure_body():
    return render_template('measure.html')

@app.route('/calculate_weight', methods=['POST'])
def calculate_weight():
    data = request.get_json()
    height = float(data['height'])
    weight = float(data['weight']) if 'weight' in data else None
    
    healthy_range = calculate_healthy_weight(height, weight)
    return jsonify({
        'lower_weight': round(healthy_range[0], 2),
        'upper_weight': round(healthy_range[1], 2)
    })

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        if frame is None:
            continue
        if exercise_tracker.is_tracking():
            frame = exercise_tracker.process_frame(frame)
            
            # Emit exercise stats via Socket.IO
            stats = exercise_tracker.get_stats()
            socketio.emit('exercise_stats', stats)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True)

