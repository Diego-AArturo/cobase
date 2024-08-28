# -------------------librerias---------------- 

import cv2
import mediapipe as mp
import numpy as np
import base64
import threading

#=========== función para cálculo de ángulo ==================
def calcular_angulo(a, b, c):
    a = np.array(a)  # primer punto
    b = np.array(b)  # punto medio
    c = np.array(c)  # punto final

    radianes = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = np.abs(radianes * 180 / np.pi)

    if angulo > 180:
        angulo = 360 - angulo

    return angulo

#==============================================
#==============================================
class PoseDetect:
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
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.processing:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    alto, ancho, _ = frame.shape
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    resul = pose.process(frame_rgb)

                    if resul.pose_landmarks is not None:
                        marca = resul.pose_landmarks.landmark

                        # Puntos de interés
                        puntos = {
                            "cd": (int(marca[14].x * ancho), int(marca[14].y * alto)),
                            "ci": (int(marca[13].x * ancho), int(marca[13].y * alto)),
                            "md": (int(marca[16].x * ancho), int(marca[16].y * alto)),
                            "mi": (int(marca[15].x * ancho), int(marca[15].y * alto)),
                            "hd": (int(marca[12].x * ancho), int(marca[12].y * alto)),
                            "hi": (int(marca[11].x * ancho), int(marca[11].y * alto)),
                            "cad": (int(marca[24].x * ancho), int(marca[24].y * alto)),
                            "cai": (int(marca[23].x * ancho), int(marca[23].y * alto)),
                            "rd": (int(marca[26].x * ancho), int(marca[26].y * alto)),
                            "ri": (int(marca[25].x * ancho), int(marca[25].y * alto)),
                            "td": (int(marca[28].x * ancho), int(marca[28].y * alto)),
                            "ti": (int(marca[27].x * ancho), int(marca[27].y * alto)),
                            "pud": (int(marca[32].x * ancho), int(marca[32].y * alto)),
                            "pui": (int(marca[31].x * ancho), int(marca[31].y * alto)),
                        }

                        # Cálculo de ángulos
                        angulos = {
                            "ang_cad": calcular_angulo(puntos["hd"], puntos["cad"], puntos["rd"]),
                            "ang_rd": calcular_angulo(puntos["cad"], puntos["rd"], puntos["td"]),
                            "ang_hd": calcular_angulo(puntos["cd"], puntos["hd"], puntos["cad"]),
                            "ang_cd": calcular_angulo(puntos["md"], puntos["cd"], puntos["hd"]),
                            "ang_cai": calcular_angulo(puntos["hi"], puntos["cai"], puntos["ri"]),
                            "ang_ri": calcular_angulo(puntos["cai"], puntos["ri"], puntos["ti"]),
                            "ang_hd_i": calcular_angulo(puntos["ci"], puntos["hi"], puntos["cai"]),
                            "ang_cd_i": calcular_angulo(puntos["mi"], puntos["ci"], puntos["hi"]),
                        }

                        # Dibujo de líneas y círculos
                        cv2.line(frame, puntos["cd"], puntos["md"], (0, 255, 0), 3)
                        cv2.line(frame, puntos["cd"], puntos["hd"], (0, 255, 0), 3)
                        cv2.line(frame, puntos["cad"], puntos["hd"], (0, 255, 0), 3)
                        cv2.line(frame, puntos["rd"], puntos["cad"], (0, 255, 0), 3)
                        cv2.line(frame, puntos["pud"], puntos["td"], (0, 255, 0), 3)

                        cv2.line(frame, puntos["ci"], puntos["mi"], (255, 0, 0), 3)
                        cv2.line(frame, puntos["ci"], puntos["hi"], (255, 0, 0), 3)
                        cv2.line(frame, puntos["cai"], puntos["hi"], (255, 0, 0), 3)
                        cv2.line(frame, puntos["ri"], puntos["cai"], (255, 0, 0), 3)
                        cv2.line(frame, puntos["pui"], puntos["ti"], (255, 0, 0), 3)

                        cv2.circle(frame, puntos["hd"], 6, (0, 0, 0), 4)
                        cv2.circle(frame, puntos["td"], 6, (0, 0, 0), 4)
                        cv2.circle(frame, puntos["cad"], 6, (0, 0, 0), 4)
                        cv2.circle(frame, puntos["rd"], 6, (0, 0, 0), 4)

                        cv2.putText(frame, str(int(angulos["ang_cad"])), (puntos["cad"][0] - 60, puntos["cad"][1]), 1, 1.5, (0, 255, 0), 2)
                        cv2.putText(frame, str(int(angulos["ang_rd"])), (puntos["rd"][0] - 60, puntos["rd"][1]), 1, 1.5, (0, 255, 0), 2)
                        cv2.putText(frame, str(int(angulos["ang_hd"])), (puntos["hd"][0] - 60, puntos["hd"][1]), 1, 1.5, (0, 255, 0), 2)
                        cv2.putText(frame, str(int(angulos["ang_cd"])), (puntos["cd"][0] - 60, puntos["cd"][1]), 1, 1.5, (0, 255, 0), 2)

                    _, buffer = cv2.imencode('.jpg', frame)
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
