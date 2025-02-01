import cv2
import multiprocessing
from flask import Flask, Response
import socketio

# Assuming exercise_tracker is already defined

# Create a multiprocessing queue to share data between processes
frame_queue = multiprocessing.Queue()

# Function to capture frames from the camera
def capture_frames(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_queue.put(frame)

# Function to process frames
def process_frames(frame_queue):
    while True:
        frame = frame_queue.get()  # Get frame from the queue
        
        if exercise_tracker.is_tracking():
            frame = exercise_tracker.process_frame(frame)
            
            # Emit exercise stats via Socket.IO
            stats = exercise_tracker.get_stats()
            socketio.emit('exercise_stats', stats)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames():
    cap = cv2.VideoCapture("http://192.168.1.3:4747/video")
    
    # Start the multiprocessing processes
    capture_process = multiprocessing.Process(target=capture_frames, args=(cap, frame_queue))
    capture_process.start()
    
    process_process = multiprocessing.Process(target=process_frames, args=(frame_queue,))
    process_process.start()

    # Join the processes to keep them running
    capture_process.join()
    process_process.join()


