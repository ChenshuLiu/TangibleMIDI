import cv2
import pygame
import mediapipe as mp
import threading
import time
import numpy as np

# Initialize Mediapipe and audio
mp_hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Initialize Pygame for audio playback
pygame.mixer.init()
pygame.mixer.music.load("/Users/chenshu/Documents/Research/Mediapipe/Sia - Unstoppable (Official Video - Live from the Nostalgic For The Present Tour).mp3")
pygame.mixer.music.play(-1)  # Loop indefinitely

# Shared variable and lock for hand landmarks
current_landmarks = None
controlled_feature = None
lock = threading.Lock()

def landmarks_to_list(landmark_list):
    """Converts a MediaPipe NormalizedLandmarkList to a Python list of landmarks."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list.landmark])

def contact_with_palm_cm(tip_coord, cm_coord, radius):
    # [x_1, y_1]
    # [x_cm, y_cm]
    interact = False
    range_dist = radius ** 2
    if range_dist >= np.sum([(cm_coord[idx] - tip_coord[idx])**2 for idx in range(len(tip_coord))]):
        interact = True
    return interact # bool (whether in contact)

# Function to control audio based on hand landmarks
def control_audio_with_landmarks(landmarks):
    if controlled_feature == 'volume':
        # adjust volume based on y-coordinate of index finger
        index_finger_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        volume = max(0, min(1, 1 - index_finger_tip.y))  # Clamp volume to 0-1
        pygame.mixer.music.set_volume(volume)
        print(f"Adjusting volume to {volume * 100:.2f}%")
    elif controlled_feature == 'reverb':
        pass
    elif controlled_feature == 'pitch': 
        middle_finger_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        pitch = max(0, min(1, 1 - middle_finger_tip.y))
        pygame.mixer.music.set_pos
    else: # when none
        pass

# Audio control thread
def audio_control():
    global current_landmarks
    while True:
        with lock:
            if current_landmarks is not None:
                control_audio_with_landmarks(current_landmarks)
        time.sleep(0.01)  # Adjusting volume every 100ms

# Start audio control in a separate thread
audio_thread = threading.Thread(target=audio_control, daemon=True)
audio_thread.start()

mp_drawing_styles = mp.solutions.drawing_styles
landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style()
# Main loop for video capture and hand landmark tracking
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame to get hand landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)

    with lock:
        if results.multi_hand_landmarks:
            # Update shared landmarks for the audio thread
            current_landmarks = results.multi_hand_landmarks[0]  # Using first detected hand

    # Draw hand landmarks on the frame for visualization
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # for cm
            radius_cm = 50
            temp_landmarks = landmarks_to_list(hand_landmarks) # numpy array with shape [21, 3]
            temp_landmarks_palm = temp_landmarks[[0, 1, 2, 5, 9, 13, 17], :]
            cm_x, cm_y, _ = np.mean(temp_landmarks, axis=0)
            cm_x_palm, cm_y_palm, _ = np.mean(temp_landmarks_palm, axis = 0)
            cm_x = int(cm_x * frame.shape[1])
            cm_y = int(cm_y * frame.shape[0])
            cm_x_palm = int(cm_x_palm * frame.shape[1])
            cm_y_palm = int(cm_y_palm * frame.shape[0])
            cv2.circle(frame, (cm_x_palm, cm_y_palm), radius=radius_cm, color = (0, 0, 255))
            print(temp_landmarks[4, :2])

            # determine which function / color to use and show (thumb, index, middle finger)
            if contact_with_palm_cm([temp_landmarks[4, 0]*frame.shape[1], temp_landmarks[4, 1]*frame.shape[0]], 
                                    [cm_x_palm, cm_y_palm], radius_cm): # when thumb is in contact with cm button
                # thumb release the current feature being controlled
                landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style()
                controlled_feature = None
            elif contact_with_palm_cm([temp_landmarks[8, 0]*frame.shape[1], temp_landmarks[8, 1]*frame.shape[0]], 
                                      [cm_x_palm, cm_y_palm], radius_cm):
                # index control the loudness of the audio
                landmark_color_index = (0, 255, 0)
                landmark_drawing_spec = mp_drawing.DrawingSpec(color=landmark_color_index, circle_radius=5)
                controlled_feature = 'volume'
            elif contact_with_palm_cm([temp_landmarks[12, 0]*frame.shape[1], temp_landmarks[12, 1]*frame.shape[0]], 
                                      [cm_x_palm, cm_y_palm], radius_cm):
                landmark_color_middle = (0, 0, 255)
                landmark_drawing_spec = mp_drawing.DrawingSpec(color=landmark_color_middle, circle_radius=5)
                controlled_feature = 'pitch'
            elif contact_with_palm_cm([temp_landmarks[4, 0]*frame.shape[1], temp_landmarks[4, 1]*frame.shape[0]],
                                      [temp_landmarks[8, 0]*frame.shape[1], temp_landmarks[8, 1]*frame.shape[0]],
                                      radius = 35): # the pitch is detected between index and thumb
                landmark_color_pinch = (255, 0, 0)
                landmark_drawing_spec = mp_drawing.DrawingSpec(color=landmark_color_pinch, circle_radius=5)
                controlled_feature = 'reverb'

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, landmark_drawing_spec)

    # Show the video frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
