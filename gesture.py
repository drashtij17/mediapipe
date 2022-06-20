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
                    print(hand_label)
            # Retrieve the landmarks of the found hand.
                    hand_landmarks =  results.multi_hand_landmarks[hand_index]
                    for tip_index in fingers_tips_ids:
                        # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
                        finger_name = tip_index.name.split("_")[0]
                        if hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y:  
                            fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                        # Increment the count of the fingers up of the hand by 1.
                        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
                        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
                        if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
                            # Update the status of the thumb in the dictionary to true.
                            fingers_statuses[hand_label.upper()+"_THUMB"] = True

                            # Increment the count of the fingers up of the hand by 1.
                            for hand_index, hand_label in enumerate(hands_labels):
                                # Initialize a variable to store the color we will use to write the hands gestures on the image.
                                # Initially it is red which represents that the gesture is not recognized.
                                color = (0, 0, 255)
                                # Check if the person is making the 'V' gesture with the hand.
                                # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
                                if fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_INDEX']:
                                    # Update the gesture value of the hand that we are iterating upon to V SIGN.
                                    hands_gestures[hand_label] = "Yoo"
                                    cv2.putText(image,hands_gestures[hand_label] ,(960, 900),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 2,cv2.LINE_AA, True)
                                    # Update the color value to green.
                                elif fingers_statuses[hand_label+'_THUMB']:
                                    hands_gestures[hand_label] = "Thumbs Up"
                                    cv2.putText(image,hands_gestures[hand_label] ,(960, 900),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 234, 0), 2,cv2.LINE_AA, True)

        cv2.imshow('Hands',cv2.flip(image,0))
        k=cv2.waitKey(1) 
        if k == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()