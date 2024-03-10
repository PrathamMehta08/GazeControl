from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import gaze
import math
import gtts
import statistics

minm = 1000
maxm = 0

curr = "YES"
duration = 0

import pygame
from gtts import gTTS
import os

left = []
right = []

calibrating_left = True
calibrating_right = False
asked_question = False
yes_no = False

said_cal = False
said_left = False
said_right = False
said_done = False
said_ask = False
said_all = False

# Set the window to fullscreen mode
cv2.namedWindow('output window', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('output window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get screen dimensions (replace with your actual screen width and height)
screen_width = 640
screen_height = 480

def speak(s):
    tts = gTTS(text=s, lang='en', slow=False)

    tts.save("audio.mp3")

    pygame.mixer.init()

    try:
        pygame.mixer.music.load("audio.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        pygame.mixer.quit()


def dist(x, y, x1, y1):
    return math.sqrt((x - x1)**2 + (y - y1)**2)

def ask_question(annotated_image):
    text_position = (int(screen_width * 0.05), int(screen_height * 0.2))
    cv2.putText(annotated_image, 'ASK QUESTION', text_position, cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 3, cv2.LINE_AA)

    return annotated_image
    
def calibrate_right(annotated_image):
    text_position = (int(screen_width * 0.05), int(screen_height * 0.2))
    cv2.putText(annotated_image, 'LOOK RIGHT', text_position, cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 3, cv2.LINE_AA)

    return annotated_image

def calibrate_left(annotated_image):
    text_position = (int(screen_width * 0.05), int(screen_height * 0.2))
    cv2.putText(annotated_image, 'LOOK LEFT', text_position, cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 3, cv2.LINE_AA)

    return annotated_image

def draw_yes_no(annotated_image, res):
    overlay = annotated_image.copy()

    cv2.line(annotated_image, (int(screen_width * 0.5), 0), (int(screen_width * 0.5), screen_height), (255, 255, 255), 5)

    if res == "YES":
        cv2.rectangle(overlay, (0, 0), (int(screen_width * 0.5), screen_height), (0, 255, 0), -1)
    else:
        cv2.rectangle(overlay, (int(screen_width * 0.5), 0), (screen_width, screen_height), (0, 0, 255), -1)

    cv2.putText(annotated_image, 'YES', (int(screen_width * 0.1), int(screen_height * 0.2)), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.putText(annotated_image, 'NO', (int(screen_width * 0.6), int(screen_height * 0.2)), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 255, 255), 3, cv2.LINE_AA)

    annotated_image = cv2.addWeighted(overlay, 0.4, annotated_image, 0.6, 0)

    return annotated_image

def get_x(minm, maxm, d):
    x = min(max((d - minm) * (screen_width) / (maxm - minm), 0), screen_width)

    return int(x)

def draw_landmarks_on_image(rgb_image, detection_result):
    global minm, maxm

    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    res = "YES"

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        d = dist(face_landmarks[468].x, face_landmarks[468].y, face_landmarks[155].x, face_landmarks[155].y)
        x = get_x(minm, maxm, d)

        if (calibrating_left): left.append(d)
        if (calibrating_right): right.append(d)
        else: cv2.circle(annotated_image, (x, 300), 10, (255, 0, 0), -1)

        if (x < 300):
            res = "YES"
        else:
            res = "NO"

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image, res

mp_face_mesh = solutions.face_mesh

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

import time
startTime = time.time()

# Camera stream:
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as face_mesh:
    while cap.isOpened():
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)

        cv2.imwrite("temp_frame.jpg", image)
        created_image = mp.Image.create_from_file("temp_frame.jpg")
        
        image.flags.writeable = False
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(created_image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            gaze.gaze(image, results.multi_face_landmarks[0])

        annotated_image, result = draw_landmarks_on_image(image, detection_result)

        if (not said_all):
            if (time.time() - startTime > 2 and time.time() - startTime < 5 and not said_cal):
                speak("Calibrating")
                speak("Follow the instructions")
                said_cal = True
            if (time.time() - startTime > 5 and time.time() - startTime < 10):
                if (not said_left):
                    speak("Look Left")
                    said_left = True
            elif time.time() - startTime > 10 and time.time() - startTime < 15:
                calibrating_left = False
                calibrating_right = True
                if (not said_right):
                    speak("Look Right")
                    said_right= True
                minm = statistics.median(left)
            elif time.time() - startTime > 15 and time.time() - startTime < 20:
                calibrating_right = False
                asked_question = True
                maxm = statistics.median(right)
                
                if (not said_done):
                    speak("Calibrating Finished")
                    said_done = True
                    
                if (not said_ask):
                    speak("What is your question")
                    said_ask = True
                    
            elif time.time() - startTime > 25:
                asked_question = False
                yes_no = True
                startTime = time.time()
                said_all = True
            

        if (calibrating_left): annotated_image = calibrate_left(annotated_image)
        if (calibrating_right): annotated_image = calibrate_right(annotated_image)
        if (asked_question): annotated_image = ask_question(annotated_image)
        if (yes_no): annotated_image = draw_yes_no(annotated_image, result)

        if (yes_no):
            if (result != curr):
                startTime = time.time()
                curr = result

            if time.time() - startTime > 2:
                speak(curr)
                startTime = time.time()
                asked_question = True
                yes_no = False

        if (asked_question and said_all):
            if (time.time() - startTime > 5):
                startTime = time.time()
                yes_no = True
                asked_question = False
            
        cv2.imshow('output window', annotated_image)

        os.remove("temp_frame.jpg")

        if cv2.waitKey(2) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
