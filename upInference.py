import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_lm = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.5, max_num_hands=4)

# Labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'G'}

# Function to process video frames
def process_frame():
    _, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use the Hand Landmark model to detect landmarks
    results_lm = hands_lm.process(frame_rgb)
    if results_lm.multi_hand_landmarks:
        for hand_landmarks in results_lm.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            label.config(text=predicted_character)

    # Update the GUI window
    window.after(10, process_frame)

# Create the GUI window
window = tk.Tk()
window.title("Hand Gesture Recognition")

# Create a label to display the recognized text
label = tk.Label(window, text="", font=("Arial", 24))
label.pack(pady=20)

# Start processing video frames
process_frame()

# Run the GUI main loop
window.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
