from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
'LEFT_RING': False, 'LEFT_PINKY': False}
fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
# Store the labels of both hands in a list.
hands_labels = ['RIGHT', 'LEFT']
# Initialize a dictionary to store the gestures of both hands in the image.
hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}
count = {'RIGHT': 0, 'LEFT': 0}
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
        multiLandMarks = results.multi_hand_landmarks
        image.flags.writeable = True
        image = cv2.cvtColor(image,COLOR_RGB2BGR)
        if multiLandMarks:
            handList = []
            for handLms in multiLandMarks:
                mp_drawing.draw_landmarks(
                image,
                handLms,
                mp_hands.HAND_CONNECTIONS,
                )
                for hand_index, hand_info in enumerate(results.multi_handedness):
            # Retrieve the label of the found hand.
                    hand_label = hand_info.classification[0].label
                    print("######",hand_index)
            # Retrieve the landmarks of the found hand.
                    hand_landmarks =  results.multi_hand_landmarks[hand_index]
                    for tip_index in fingers_tips_ids:
                        # print("^&&&&&&&&",tip_index)
                        # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
                        finger_name = tip_index.name.split("_")[0]
                        # print("@@",hand_landmarks.landmark[tip_index].y)
                        # print("!!!!!",hand_landmarks.landmark[tip_index - 2].y)
                        if hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y:  
                            fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                            # print("$$$$$$$",fingers_statuses)
                            # print("Name",finger_name)
                            # print("^^^^^^^",fingers_statuses[hand_label.upper()+"_"+finger_name])
                            count[hand_label.upper()]+=1
                    print(count)
                        # Increment the count of the fingers up of the hand by 1.
                    thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                    thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
                        
                    if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
                    
                        fingers_statuses[hand_label+"_THUMB"] = True
                            
                        for hand_index, hand_label in enumerate(hands_labels):
                            color = (0, 0, 255)
                            if fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label.upper()+'_INDEX']:
                                hands_gestures[hand_label] = "Yoo"
                                cv2.putText(image,hands_gestures[hand_label] ,(960, 900),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 2,cv2.LINE_AA, True)
                                
                            if fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_PINKY']:
                                hands_gestures[hand_label] = "Cool"
                                cv2.putText(image,hands_gestures[hand_label] ,(960, 900),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 234, 0), 2,cv2.LINE_AA, True)

        cv2.imshow('Hands',cv2.flip(image,0))
        k=cv2.waitKey(1) 
        if k == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
