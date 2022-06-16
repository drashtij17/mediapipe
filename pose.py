from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    smooth_landmarks = True,
    smooth_segmentation = True,
    # refine_face_landmarks = True,
    model_complexity = 0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    ) as holistic:
    while True:
        rel,image = cap.read()
        image.flags.writeable = False
        image = cv2.cvtColor(image,COLOR_BGR2RGB)
        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image,COLOR_RGB2BGR)
        
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.face_landmarks,
        #     connections = mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_contours_style()
        # )
       
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_holistic.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles
        #     .get_default_pose_landmarks_style()
        # )
        
        mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        )
        mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        )
        print(results.left_hand_landmarks)
        if results.left_hand_landmarks:
            print("its is left hand.......")
    
        cv2.imshow('Image',cv2.flip(image,0))
        k=cv2.waitKey(1) 
        if k == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()