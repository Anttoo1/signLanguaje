
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Cargar el modelo
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Diccionario de etiquetas
labels_dict = {0: 'A', 1: 'B', 2: 'C', 4: 'A'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Usar el modelo de landmarks de MediaPipe Hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # imagen en la que dibujar
                hand_landmarks,  # salida del modelo
                mp_hands.HAND_CONNECTIONS,  # conexiones de la mano
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

            all_landmarks.extend(data_aux)

        if len(all_landmarks) == 84:  # 21 landmarks por mano * 2 coordenadas * 2 manos = 84
            prediction = model.predict([np.asarray(all_landmarks)])
            predicted_character = labels_dict[int(prediction[0])]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Ajustar las coordenadas del rectángulo para abarcar ambas manos
            for landmarks in results.multi_hand_landmarks:
                for landmark in landmarks.landmark:
                    x = int(landmark.x * W)
                    y = int(landmark.y * H)
                    if x < x1:
                        x1 = x
                    if x > x2:
                        x2 = x
                    if y < y1:
                        y1 = y
                    if y > y2:
                        y2 = y

            # Cambiar el color del rectángulo (en este caso, azul)
            color = (255, 0, 0)  # Azul en formato BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
