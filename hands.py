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

                    results = hands.process(image_rgb)

                    if results.multi_hand_landmarks is not None:
                        for marks in results.multi_hand_landmarks:
                            # Obtener la posición del dedo índice
                            index_finger_tip_x = int(marks.landmark[8].x * width)
                            index_finger_tip_y = int(marks.landmark[8].y * heigth)

                            # Aquí se enviarían las coordenadas al cliente
                            self.page.send({
                                "type": "update_cursor",
                                "x": index_finger_tip_x,
                                "y": index_finger_tip_y,
                            })

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

