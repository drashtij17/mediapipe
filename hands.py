from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR
import mediapipe as mp
import cv2
from google.protobuf.json_format import MessageToDict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)as hands:

    while True:
        rel,image = cap.read()
        image.flags.writeable = False
        image = cv2.cvtColor(image,COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image,COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hands_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image,
                hands_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            print(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST])
                # print(type(results.multi_handedness), "=======", type(results.multi_handedness[0]))
                # for i in results.multi_handedness:
                #     label = MessageToDict(i)['classification'][0]['label']
                #     if label == 'Left':
                #         cv2.putText(image,label+' Hand',(960, 900),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0,255),2,cv2.LINE_AA, True)
                #     elif label == "Right":
                #          cv2.putText(image,label+' Hand',(960,900),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0,255),2,cv2.LINE_AA, True)
                #     else:
                #         print("not found")

            # for i in results.multi_handedness:

            #     print("") 
        cv2.imshow('Hands',cv2.flip(image,1))
        k=cv2.waitKey(1) 
        if k == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


# {<HandLandmark.WRIST: 0>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5),
#  <HandLandmark.THUMB_CMC: 1>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5), 
#  <HandLandmark.INDEX_FINGER_MCP: 5>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5),
#   <HandLandmark.MIDDLE_FINGER_MCP: 9>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5), 
#   <HandLandmark.RING_FINGER_MCP: 13>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5), 
#   <HandLandmark.PINKY_MCP: 17>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5), 
#   <HandLandmark.THUMB_MCP: 2>: DrawingSpec(color=(180, 229, 255), thickness=-1, circle_radius=5), 
#   <HandLandmark.THUMB_IP: 3>: DrawingSpec(color=(180, 229, 255), thickness=-1, circle_radius=5), 
#   <HandLandmark.THUMB_TIP: 4>: DrawingSpec(color=(180, 229, 255), thickness=-1, circle_radius=5), 
#   <HandLandmark.INDEX_FINGER_PIP: 6>: DrawingSpec(color=(128, 64, 128), thickness=-1, circle_radius=5), 
# <HandLandmark.INDEX_FINGER_DIP: 7>: DrawingSpec(color=(128, 64, 128), thickness=-1, circle_radius=5), 
# <HandLandmark.INDEX_FINGER_TIP: 8>: DrawingSpec(color=(128, 64, 128), thickness=-1, circle_radius=5), 
# <HandLandmark.MIDDLE_FINGER_PIP: 10>: DrawingSpec(color=(0, 204, 255), thickness=-1, circle_radius=5),
#  <HandLandmark.MIDDLE_FINGER_DIP: 11>: DrawingSpec(color=(0, 204, 255), thickness=-1, circle_radius=5)
# , <HandLandmark.MIDDLE_FINGER_TIP: 12>: DrawingSpec(color=(0, 204, 255), thickness=-1, circle_radius=5), 
# <HandLandmark.RING_FINGER_PIP: 14>: DrawingSpec(color=(48, 255, 48), thickness=-1, circle_radius=5),
#  <HandLandmark.RING_FINGER_DIP: 15>: DrawingSpec(color=(48, 255, 48), thickness=-1, circle_radius=5),
#   <HandLandmark.RING_FINGER_TIP: 16>: DrawingSpec(color=(48, 255, 48), thickness=-1, circle_radius=5),
#    <HandLandmark.PINKY_PIP: 18>: DrawingSpec(color=(192, 101, 21), thickness=-1, circle_radius=5),
#     <HandLandmark.PINKY_DIP: 19>: DrawingSpec(color=(192, 101, 21), thickness=-1, circle_radius=5), 
# <HandLandmark.PINKY_TIP: 20>: DrawingSpec(color=(192, 101, 21), thickness=-1, circle_radius=5)}

# {<HandLandmark.WRIST: 0>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5), 
# <HandLandmark.THUMB_CMC: 1>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5),
#  <HandLandmark.INDEX_FINGER_MCP: 5>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5), 
#  <HandLandmark.MIDDLE_FINGER_MCP: 9>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5),
#   <HandLandmark.RING_FINGER_MCP: 13>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5),
#    <HandLandmark.PINKY_MCP: 17>: DrawingSpec(color=(48, 48, 255), thickness=-1, circle_radius=5), 
#    <HandLandmark.THUMB_MCP: 2>: DrawingSpec(color=(180, 229, 255), thickness=-1, circle_radius=5),
#     <HandLandmark.THUMB_IP: 3>: DrawingSpec(color=(180, 229, 255), thickness=-1, circle_radius=5), 
#     <HandLandmark.THUMB_TIP: 4>: DrawingSpec(color=(180, 229, 255), thickness=-1, circle_radius=5),
#      <HandLandmark.INDEX_FINGER_PIP: 6>: DrawingSpec(color=(128, 64, 128), thickness=-1, circle_radius=5), 
#      <HandLandmark.INDEX_FINGER_DIP: 7>: DrawingSpec(color=(128, 64, 128), thickness=-1, circle_radius=5),
#       <HandLandmark.INDEX_FINGER_TIP: 8>: DrawingSpec(color=(128, 64, 128), thickness=-1, circle_radius=5), 
#       <HandLandmark.MIDDLE_FINGER_PIP: 10>: DrawingSpec(color=(0, 204, 255), thickness=-1, circle_radius=5),
#        <HandLandmark.MIDDLE_FINGER_DIP: 11>: DrawingSpec(color=(0, 204, 255), thickness=-1, circle_radius=5),
#         <HandLandmark.MIDDLE_FINGER_TIP: 12>: DrawingSpec(color=(0, 204, 255), thickness=-1, circle_radius=5),
#          <HandLandmark.RING_FINGER_PIP: 14>: DrawingSpec(color=(48, 255, 48), thickness=-1, circle_radius=5), 
#          <HandLandmark.RING_FINGER_DIP: 15>: DrawingSpec(color=(48, 255, 48), thickness=-1, circle_radius=5),
#           <HandLandmark.RING_FINGER_TIP: 16>: DrawingSpec(color=(48, 255, 48), thickness=-1, circle_radius=5),
#            <HandLandmark.PINKY_PIP: 18>: DrawingSpec(color=(192, 101, 21), thickness=-1, circle_radius=5), 
#            <HandLandmark.PINKY_DIP: 19>: DrawingSpec(color=(192, 101, 21), thickness=-1, circle_radius=5),
#             <HandLandmark.PINKY_TIP: 20>: DrawingSpec(color=(192, 101, 21), thickness=-1, circle_radius=5)}

