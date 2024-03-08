from django.shortcuts import render
from django.http import HttpResponse
import cv2
import numpy as np
import os
import time
import mediapipe as mp
import tensorflow as tf
import pyttsx3
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse

# Load the model
#############

############
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30
# Folder start
model = tf.keras.models.load_model('base/action.h5')
############
def process_video(video_path):
    sequence = []
    sentence = []
    outputList = []
    predictions = np.array([[]])
    threshold = 0.5

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame_number in range(total_frames):
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
                    
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Show to screen (optional)
            cv2.imshow('OpenCV Feed', image)

        # Process all frames at once
        predictions = np.array(sequence).reshape(-1, 1662)
        chunk_size = 30
        for i in range(0, len(predictions), chunk_size):
            chunk = predictions[i:i + chunk_size]
            if chunk.shape[0] == 30:
                res = model.predict(np.expand_dims(chunk, axis=0))[0]
                outputList.append(actions[np.argmax(res)])
                # live_text_to_speech(actions[np.argmax(res)])

    cap.release()
    cv2.destroyAllWindows()

    response_data = {
        'success': True,
        'message': 'Video processed successfully.',
        'output_list': outputList,
    }
    return response_data

############
# Optionally, you can check the model summary

##########
def live_text_to_speech(text, rate=150):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set speech rate (words per minute)
    engine.setProperty('rate', rate)

    # Speak the given text
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()
#########


        


######
@api_view(['POST'])
def home(request):
    if request.method == 'POST':
        try:
            video_data = request.data
            script_directory = os.path.dirname(os.path.abspath(__file__))
            video_file_path = os.path.join(script_directory, video_data['video_file'].name)

            with open(video_file_path, 'wb') as video_file:
                for chunk in video_data['video_file'].chunks():
                    video_file.write(chunk)

            processed_data = process_video(video_file_path)

            return JsonResponse({'success': True, 'message': 'Video processed successfully.', 'data': processed_data})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=500)

    return JsonResponse({'success': False, 'message': 'Invalid request method.'}, status=400)