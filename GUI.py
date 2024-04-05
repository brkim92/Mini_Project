import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox

max_num_hands = 2
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
    
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5)    

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)    

# Initialize player bounding boxes
player1_bbox = None
player2_bbox = None

player1_score = 0
player2_score = 0
score_pause_time = 0

def save_player_images(player1_face, player2_face, cap):
    cv2.imwrite('player1.jpg', cv2.cvtColor(player1_face, cv2.COLOR_RGB2BGR))
    cv2.imwrite('player2.jpg', cv2.cvtColor(player2_face, cv2.COLOR_RGB2BGR))
    messagebox.showinfo("Image Saved", "Player images saved successfully!")

    # 웹캠 종료
    cap.release()
    cv2.destroyAllWindows()


def detect_players(img, cap):
    global player1_score, player2_score, score_pause_time, player1_bbox, player2_bbox

    results = face_detection.process(img)
    if results.detections:
        for i, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Assuming the first detected face is Player 1 and the second is Player 2
            player1_bbox = bbox if i == 0 else player1_bbox
            player2_bbox = bbox if i == 1 else player2_bbox

            # Draw rectangles around players
            cv2.rectangle(img, (player1_bbox[0], player1_bbox[1]), (player1_bbox[0]+player1_bbox[2], player1_bbox[1]+player1_bbox[3]), (0, 255, 0), 2)
            if player2_bbox:
                cv2.rectangle(img, (player2_bbox[0], player2_bbox[1]), (player2_bbox[0]+player2_bbox[2], player2_bbox[1]+player2_bbox[3]), (0, 255, 255), 2)

                # Display Player 1 or Player 2 text based on face index
                player_text_color = (0, 0, 255) if i == 0 else (255, 0, 0)
                player_text = f'Player {i + 1}'
                org = (int((player1_bbox[0] + player2_bbox[0]) / 2), int((player1_bbox[1] + player2_bbox[1]) / 2))
                cv2.putText(img, text=player_text, org=org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=player_text_color, thickness=2)

                # Save images
                player1_face = img[player1_bbox[1]:player1_bbox[1]+player1_bbox[3], player1_bbox[0]:player1_bbox[0]+player1_bbox[2]]
                player2_face = img[player2_bbox[1]:player2_bbox[1]+player2_bbox[3], player2_bbox[0]:player2_bbox[0]+player2_bbox[2]]
                save_player_images(player1_face, player2_face, cap)

    return img


    
def start_hand_detection():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 가로 해상도를 640으로 설정
    cap.set(4, 480)  # 세로 해상도를 480으로 설정

    global player1_score, player2_score, score_pause_time, player1_bbox, player2_bbox
    player1_score = 0
    player2_score = 0
    score_pause_time = 0

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = detect_hands(img)

        cv2.imshow('Game', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_hands(img):
    result = hands.process(img)
    global player1_score, player2_score, score_pause_time
    # Display player scores
    cv2.putText(img, text=f'Player 1: {player1_score}', org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    cv2.putText(img, text=f'Player 2: {player2_score}', org=(img.shape[1] - 220, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    if result.multi_hand_landmarks is not None:
        rps_result = []

        for i, res in enumerate(result.multi_hand_landmarks):
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]  # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]  # Child joint
            v = v2 - v1  # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

                # Display Player 1 or Player 2 text based on hand index
                player_text_color = (0, 0, 255) if i == 0 else (255, 0, 0)
                player_text = f'Player {i + 1}'
                cv2.putText(img, text=player_text, org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=player_text_color, thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        # Who wins?
        if len(rps_result) >= 2:
            winner = None
            text = ''

            if rps_result[0]['rps'] == 'rock':
                if rps_result[1]['rps'] == 'rock': text = 'Replay'
                elif rps_result[1]['rps'] == 'paper': text = 'Paper wins'; winner = 2
                elif rps_result[1]['rps'] == 'scissors': text = 'Rock wins'; winner = 1
            elif rps_result[0]['rps'] == 'paper':
                if rps_result[1]['rps'] == 'rock': text = 'Paper wins'; winner = 1
                elif rps_result[1]['rps'] == 'paper': text = 'Replay'
                elif rps_result[1]['rps'] == 'scissors': text = 'Scissors wins'; winner = 2
            elif rps_result[0]['rps'] == 'scissors':
                if rps_result[1]['rps'] == 'rock': text = 'Rock wins'; winner = 2
                elif rps_result[1]['rps'] == 'paper': text = 'Scissors wins'; winner = 1
                elif rps_result[1]['rps'] == 'scissors': text = 'Replay'

            if winner is not None:
                cv2.putText(img, text='Winner', org=(int(img.shape[1] / 4), 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 255, 0), thickness=3)
                if time.time() > score_pause_time:
                    player1_score += 1 if winner == 1 and time.time() > score_pause_time else 0
                    player2_score += 1 if winner == 2 and time.time() > score_pause_time else 0
                    score_pause_time = time.time() + 2  # Pause the score for 1 second

                    # Increment player score and reset if 10 points are reached
                    if player1_score == 10 or player2_score == 10:
                        winner_text = f'Player {winner} Wins!'
                        cv2.putText(img, text=winner_text, org=(int(img.shape[1] / 3), 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                        cv2.imshow('Game', img)
                        player1_score = 0
                        player2_score = 0
                        score_pause_time = time.time() + 5  # Pause the score for 5 second

            cv2.putText(img, text=text, org=(int(img.shape[1] / 4), 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(148, 0, 211), thickness=2)

    return img
    
def start_game(mode):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 가로 해상도를 640으로 설정
    cap.set(4, 480)  # 세로 해상도를 480으로 설정

    global player1_score, player2_score, score_pause_time
    player1_score = 0
    player2_score = 0
    score_pause_time = 0

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if mode == 1:
            img = detect_hands(img)

        elif mode == 2:
            img = detect_players(img, cap)  # cap 변수를 전달
            

        
        cv2.imshow('Game', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI 생성
def create_start_window():
    start_window = tk.Tk()
    start_window.title("Rock Scssiors Paper")

    # Face Detection 버튼
    face_button = tk.Button(start_window, text="플레이어 등록", command=lambda: start_game(mode=2))
    face_button.pack(pady=10)

    # Hand Detection 버튼
    hand_button = tk.Button(start_window, text="Hand Detection", command=lambda: start_game(mode=1))
    hand_button.pack(pady=10)

    start_window.mainloop()

# 시작 창 열기
create_start_window()

