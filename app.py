from multiprocessing import connection
from sre_constants import SUCCESS
import cv2
from cv2 import COLOR_BGR2RGB
from cv2 import COLOR_RGB2BGR
import mediapipe as mp
from pkg_resources import get_default_cache

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while True :
        ret,image = cap.read()
        cv2.imshow("face mesh",image)
        k=cv2.waitKey(1)
        image.flags.writeable = False
        image = cv2.cvtColor(image,COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image,COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # border around face,iris and lips
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                print(mp_face_mesh.FACEMESH_IRISES)
        cv2.imshow('Image',cv2.flip(image,0))
        if k==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# if cv2.waitKey(1) & 0xFF == ord('q'):