import threading
import cv2
import sounddevice as sd
import librosa
import numpy as np
import time
import mediapipe as mp
import math
from pydub import AudioSegment
from collections import deque

# Load and prepare audio
audio, sr = librosa.load("./Sia - Unstoppable.mp3", sr=None)
volume_factor = 1.0  # controlling volume level
pitch_factor = 0 # controlling pitch level (not shifting the pitch at all) 4 steps is a major third
reverb_factor = 0 # overlaying (looping to add delayed chunks)

echo1_delay_ms = 0.1 # delay 20ms
echo1_volume_level = 0.6
echo1 = np.concatenate((np.zeros(int(echo1_delay_ms*sr)), audio))[:len(audio)]
echo2_delay_ms = 0.2 # delay 20ms
echo2_volume_level = 0.4
echo2 = np.concatenate((np.zeros(int(echo2_delay_ms*sr)), audio))[:len(audio)]
echo3_delay_ms = 0.25 # delay 20ms
echo3_volume_level = 0.2
echo3 = np.concatenate((np.zeros(int(echo3_delay_ms*sr)), audio))[:len(audio)]

echo_switch = deque(maxlen=2)
echo_switch.append(reverb_factor)
echo_switch.append(reverb_factor)

# thread management
stop_threads = False
audio_position = 0
lock = threading.Lock()
current_landmark = None  # To store landmark information
controlled_feature = None # features that can be controlled (volume, pitch level, reverb significance)

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

# def distance_between_two_points(coord1, coord2): # for determining the openness between the two fingers
#     # coords are [x, y]
#     dist_sq = np.sum([(coord1[idx] - coord2[idx]) ** 2 for idx in range(len(coord1))])
#     dist_sqrt = math.sqrt(dist_sq)
#     return dist_sqrt

# def pinch_detection(coord1, coord2, radius):
#     dist_sqrt = math.sqrt(np.sum([(coord1[idx] - coord2[idx])**2 for idx in range(len(coord1))]))
#     if dist_sqrt <= radius:
#         pinch = True
#     else:
#         pinch = False

# Audio playback function
def play_audio():
    global stop_threads, volume_factor, pitch_factor
    
    def callback(outdata, frames, time, status):
        nonlocal audio_position
        if status:
            print(status)
        
        # Apply volume factor in a thread-safe way
        with lock:
            current_volume = volume_factor  # Safely read the volume factor
            current_pitch = pitch_factor

        chunk = audio[audio_position:audio_position + frames] * current_volume # changing according to current volume level
        chunk = librosa.effects.pitch_shift(y=chunk, sr=sr, n_steps=current_pitch) # changing according to current pitch level
        audio_position += frames

        # End playback if at the end of the audio data
        if len(chunk) < frames:
            outdata[:len(chunk)] = chunk[:, np.newaxis]
            outdata[len(chunk):] = 0
            raise sd.CallbackStop
        else:
            outdata[:] = chunk[:, np.newaxis]

    def echo_callback(outdata, frames, time, status):
        global reverb_factor
        nonlocal audio_position 
        if status:
            print(status)
        
        if reverb_factor:
            with lock:
                current_volume = volume_factor  # Safely read the volume factor
                current_pitch = pitch_factor
            echo_audio_position = audio_position - frames
            echo1_chunk = echo1[echo_audio_position:echo_audio_position + frames] * current_volume * echo1_volume_level
            echo1_chunk = librosa.effects.pitch_shift(y=echo1_chunk, sr=sr, n_steps=current_pitch)
            echo2_chunk = echo2[echo_audio_position:echo_audio_position + frames] * current_volume * echo2_volume_level
            echo2_chunk = librosa.effects.pitch_shift(y=echo2_chunk, sr=sr, n_steps=current_pitch)
            echo3_chunk = echo3[echo_audio_position:echo_audio_position + frames] * current_volume * echo3_volume_level
            echo3_chunk = librosa.effects.pitch_shift(y=echo3_chunk, sr=sr, n_steps=current_pitch)
            combined_chunk = echo1_chunk + echo2_chunk + echo3_chunk

            # End playback if at the end of the audio data
            if len(combined_chunk) < frames:
                outdata[:len(combined_chunk)] = combined_chunk[:, np.newaxis]
                outdata[len(combined_chunk):] = 0
                raise sd.CallbackStop
            else:
                outdata[:] = combined_chunk[:, np.newaxis]

        else:
            outdata[:] = 0

    # Set the initial position for audio
    audio_position = 0
    with sd.OutputStream(samplerate=sr, channels=1, callback=callback) as main_audio_stream, \
        sd.OutputStream(samplerate=sr, channels=1, callback=echo_callback) as echo_audio_stream:
        while not stop_threads:
            time.sleep(0.01)


# Video processing and landmark detection function
def process_video():
    global current_landmark, controlled_feature, stop_threads
    global volume_factor, pitch_factor, reverb_factor # for displaying the values
    textcoord = None
    tag_message = None
    textcolor = None

    mp_hands = mp.solutions.hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

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
                current_landmark = results.multi_hand_landmarks[0]  # Using first detected hand
            
            current_pitch = pitch_factor
            current_echo = reverb_factor

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
                #print(temp_landmarks[4, :2])

                # determine which function / color to use and show (thumb, index, middle finger)
                if contact_with_palm_cm([temp_landmarks[4, 0]*frame.shape[1], temp_landmarks[4, 1]*frame.shape[0]], 
                                        [cm_x_palm, cm_y_palm], radius_cm): # when thumb is in contact with cm button
                    # thumb release the current feature being controlled
                    landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style()
                    controlled_feature = None
                    tag_message = "No feature controlled"
                    textcoord = 4
                    textcolor = (0, 0, 0)
                elif contact_with_palm_cm([temp_landmarks[8, 0]*frame.shape[1], temp_landmarks[8, 1]*frame.shape[0]], 
                                        [cm_x_palm, cm_y_palm], radius_cm):
                    # index control the loudness of the audio
                    landmark_color_index = (0, 255, 0)
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=landmark_color_index, circle_radius=5)
                    controlled_feature = 'volume'
                    with lock: current_volume = volume_factor
                    tag_message = f"Volume adjusted"
                    textcoord = 8
                    textcolor = landmark_color_index
                elif contact_with_palm_cm([temp_landmarks[12, 0]*frame.shape[1], temp_landmarks[12, 1]*frame.shape[0]], 
                                        [cm_x_palm, cm_y_palm], radius_cm):
                    landmark_color_middle = (0, 0, 255)
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=landmark_color_middle, circle_radius=5)
                    controlled_feature = 'pitch'
                    textcoord = 12
                    tag_message = f"Pitch adjusted"
                    textcolor = landmark_color_middle
                elif contact_with_palm_cm([temp_landmarks[4, 0]*frame.shape[1], temp_landmarks[4, 1]*frame.shape[0]],
                                        [temp_landmarks[8, 0]*frame.shape[1], temp_landmarks[8, 1]*frame.shape[0]],
                                        radius = 30): # the pitch is detected between index and thumb
                    landmark_color_pinch = (255, 0, 0)
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=landmark_color_pinch, circle_radius=5)
                    controlled_feature = 'reverb'
                    textcoord = 4
                    tag_message = f"Reverb"
                    textcolor = landmark_color_pinch
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, landmark_drawing_spec)

                if textcoord is not None and textcolor is not None:
                    cv2.putText(frame, tag_message, (int(temp_landmarks[textcoord, 0]*frame.shape[1]), 
                                                                int(temp_landmarks[textcoord, 1]*frame.shape[0] + 100)), 
                                                                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                                                fontScale = 0.5,
                                                                color = textcolor,
                                                                thickness=1,
                                                                lineType = cv2.LINE_AA)
        # Show the video frame
        cv2.imshow("Hand Tracking", frame)
        

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            stop_threads = True
            break
    cap.release()
    cv2.destroyAllWindows()

# Control function to adjust audio based on landmarks
def control_audio():
    global current_landmark, volume_factor, stop_threads, pitch_factor, reverb_factor, echo_switch, controlled_feature
    while not stop_threads:
        # Read landmark data safely
        with lock:
        # Example: Adjust volume based on the x-coordinate of a landmark
            if current_landmark is not None:
                if controlled_feature == 'volume':
                    # adjust volume based on y-coordinate of index finger
                    index_finger_tip = current_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    volume_factor = max(0, min(1, 1 - index_finger_tip.y))  # Clamp volume to 0-1
                    print(f"Adjusting volume to {volume_factor * 100:.2f}%")
                elif controlled_feature == 'reverb': 
                    # index_finger_tip = current_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    # thumb_finger_tip = current_landmark.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP] # todo: find out the type of the coordinates (are they lists?)
                    # dist_index_middle = distance_between_two_points([index_finger_tip.x, index_finger_tip.y], 
                    #                                                 [thumb_finger_tip.x, thumb_finger_tip.y]) * 10
                    # reverb_factor = pinch_detection([index_finger_tip.x, index_finger_tip.y], 
                    #                                 [thumb_finger_tip.x, thumb_finger_tip.y], 20)  # binary echo (single echo)
                    # echo_switch.append(not echo_switch[1])
                    # reverb_factor = echo_switch[1]

                    reverb_factor = not reverb_factor
                    print(f"reverb factor {reverb_factor}")
                    controlled_feature = None # because this is a one time operation, need to exit to ground everytime detect a pinch
                elif controlled_feature == 'pitch': 
                    middle_finger_tip = current_landmark.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
                    pitch_factor = int(math.floor((1-middle_finger_tip.y) * 10)/4) # clipping 0-9
                    print(f"Adjusting pitch by {pitch_factor} steps")
                else: # when none of feature specified
                    pass
        time.sleep(0.5) # check for hand updates every 100ms

# Set up and start threads
audio_thread = threading.Thread(target=play_audio)
control_thread = threading.Thread(target=control_audio)

control_thread.start()
audio_thread.start()

process_video()

# Todo: how about making the trajectory stay for a while, for example, I writing a word to construct a piece of melody