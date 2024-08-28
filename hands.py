# -------------------librerias---------------- 

import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
import pyautogui


#=========== función para trasladar coordenadas==================
def cam_to_screen(x: int, y: int, image):
    '''
    Función para trasladar las coordenadas dadas en por el video 
    a las dimensiones reales de la pantalla
    '''
    width_s, heigth_s = pyautogui.size()
    heigth, width, _ = image.shape

    y_n = (y * heigth_s) / heigth
    x_n = (x * width_s) / width

    return (x_n, y_n)

#==============================================
#==============================================
class HandDetect():
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
                            wrist_x = int(marks.landmark[0].x * width)
                            wrist_y = int(marks.landmark[0].y * heigth)
                            thumbtip_x = int(marks.landmark[4].x * width)
                            thumbtip_y = int(marks.landmark[4].y * heigth)
                            index_fin_tip_x = int(marks.landmark[8].x * width)
                            index_fin_tip_y = int(marks.landmark[8].y * heigth)
                            middle_fin_tip_x = int(marks.landmark[12].x * width)
                            middle_fin_tip_y = int(marks.landmark[12].y * heigth)

                            cv2.circle(image, (wrist_x, wrist_y), 3, (255, 0, 0), 3)
                            cv2.circle(image, (thumbtip_x, thumbtip_y), 3, (255, 0, 0), 3)
                            cv2.circle(image, (index_fin_tip_x, index_fin_tip_y), 3, (255, 0, 0), 3)
                            cv2.circle(image, (middle_fin_tip_x, middle_fin_tip_y), 3, (255, 0, 0), 3)

                            x, y = cam_to_screen(index_fin_tip_x, index_fin_tip_y, image)
                            pyautogui.moveTo(x, y)

                            rest_index_x = abs(index_fin_tip_x - thumbtip_x)
                            rest_index_y = abs(index_fin_tip_y - thumbtip_y)
                            rest_mid_x = abs(middle_fin_tip_x - thumbtip_x)
                            rest_mid_y = abs(middle_fin_tip_y - thumbtip_y)

                            if rest_index_x < 10 and rest_index_y < 15:
                                pyautogui.leftClick()

                            if rest_mid_x < 10 and rest_mid_y < 10:
                                pyautogui.rightClick()

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

