import cv2
import mediapipe as mp
import numpy as np
import base64
import threading


class HandDetect:
    def __init__(self, stream, page):
        self.cap = cv2.VideoCapture(0)
        self.streaming = True
        self.page = page
        self.stream = stream
        self.processing = False
        self.drawing = False  # Estado de dibujo
        self.erasing = False  # Estado de borrado
        self.last_x, self.last_y = None, None  # Última posición del dedo índice
        self.canvas = None  # Imagen para persistir el dibujo

    def video(self):
        while self.streaming:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                _, buffer = cv2.imencode('.jpg', frame)
                self.stream.src_base64 = base64.b64encode(buffer).decode('utf-8')
                self.page.update()

    def video_detect(self):
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while self.processing:
                ret, image = self.cap.read()
                if ret:
                    heigth, width, _ = image.shape
                    image = cv2.flip(image, 1)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Inicializar la imagen del canvas si no está creada
                    if self.canvas is None:
                        self.canvas = np.zeros_like(image)

                    results = hands.process(image_rgb)

                    if results.multi_hand_landmarks is not None:
                        for marks in results.multi_hand_landmarks:
                            # Coordenadas del dedo índice
                            index_finger_tip_x = int(marks.landmark[8].x * width)
                            index_finger_tip_y = int(marks.landmark[8].y * heigth)

                            # Coordenadas del dedo medio
                            middle_finger_tip_x = int(marks.landmark[12].x * width)
                            middle_finger_tip_y = int(marks.landmark[12].y * heigth)

                            # Detección para borrar cuando el dedo índice y el medio están juntos
                            distance_between_fingers = np.sqrt(
                                (index_finger_tip_x - middle_finger_tip_x) ** 2 +
                                (index_finger_tip_y - middle_finger_tip_y) ** 2
                            )
                            
                            # Cuando los dedos están juntos, activar el modo borrador
                            print('dis',distance_between_fingers)
                            if distance_between_fingers < 70:
                                self.erasing = True
                                self.drawing = False
                            else:
                                self.erasing = False

                            if self.erasing:
                                cv2.circle(self.canvas, (index_finger_tip_x, index_finger_tip_y), 20, (0, 0, 0), thickness=-1)
                            elif self.drawing:
                                if self.last_x is not None and self.last_y is not None:
                                    cv2.line(self.canvas, (self.last_x, self.last_y), (index_finger_tip_x, index_finger_tip_y), (0, 255, 0), 5)
                                self.last_x, self.last_y = index_finger_tip_x, index_finger_tip_y
                            else:
                                self.last_x, self.last_y = None, None

                            # Detección para empezar a dibujar (por ejemplo, usando el pulgar e índice juntos)
                            thumb_tip_x = int(marks.landmark[4].x * width)
                            thumb_tip_y = int(marks.landmark[4].y * heigth)

                            distance_thumb_index = np.sqrt(
                                (index_finger_tip_x - thumb_tip_x) ** 2 +
                                (index_finger_tip_y - thumb_tip_y) ** 2
                            )
                            
                            if distance_thumb_index > 50 and not self.erasing:
                                self.drawing = False
                            else:
                                self.drawing = True

                    # Superponer el canvas sobre la imagen original
                    image = cv2.addWeighted(image, 1, self.canvas, 1, 0)

                    _, buffer = cv2.imencode('.jpg', image)
                    self.stream.src_base64 = base64.b64encode(buffer).decode('utf-8')
                    self.page.update()

    def play_video(self):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.processing = False
        threading.Thread(target=self.video, daemon=True).start()

    def play_detect(self):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.processing = True
        self.streaming = False
        threading.Thread(target=self.video_detect, daemon=True).start()

    def stop(self):
        self.processing = False
        self.streaming = False        
        self.cap.release()
        self.stream.src_base64 = ""
        self.page.update()

    def pause(self):
        self.processing = False        
        self.streaming = True



