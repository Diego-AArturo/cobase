import cv2
import mediapipe as mp
import base64
import threading

class FaceMeshDetect():
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
        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            while self.processing:
                ret, image = self.cap.read()
                if ret:
                    image.flags.writeable = False
                    image = cv2.flip(image, 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp.solutions.drawing_styles
                                .get_default_face_mesh_tesselation_style())
                            mp.solutions.drawing_utils.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp.solutions.drawing_styles
                                .get_default_face_mesh_contours_style())
                            mp.solutions.drawing_utils.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_IRISES,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp.solutions.drawing_styles
                                .get_default_face_mesh_iris_connections_style())
                    
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

