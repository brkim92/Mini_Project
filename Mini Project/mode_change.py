import cv2
import mediapipe as mp
import numpy as np
import time

max_num_hands = 2
THRESHOLD = 0.2  # 20%, 값이 클수록 손이 카메라와 가까워야 인식함
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}
gesture = {
    0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 가로 해상도를 640으로 설정
cap.set(4, 480)  # 세로 해상도를 480으로 설정

mode = 0  # 초기 모드 선택 (0: 기본, 1: Rock, Scissors, Paper, 2: Gesture)

player1_score = 0
player2_score = 0
detect_hands = True
score_pause_time = 0

while cap.isOpened():

    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if detect_hands and mode == 1:  # 모드가 1인 경우에만 실행
        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])

                # Draw gesture result
                if idx in rps_gesture.keys():
                    org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                    cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] - 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    rps_result.append({
                        'rps': rps_gesture[idx],
                        'org': org
                    })

                    # Display Player 1 or Player 2 text based on hand index
                    player_text_color = (0, 0, 255) if i == 0 else (255, 0, 0)
                    player_text = f'Player {i + 1}'
                    cv2.putText(img, text=player_text, org=(org[0], org[1] + 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=player_text_color,
                                thickness=2)

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
                    cv2.putText(img, text='Winner', org=(int(img.shape[1] / 4), 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=3, color=(0, 255, 0), thickness=3)
                    if time.time() > score_pause_time:
                        player1_score += 1 if winner == 1 and time.time() > score_pause_time else 0
                        player2_score += 1 if winner == 2 and time.time() > score_pause_time else 0
                        score_pause_time = time.time() + 2  # Pause the score for 1 second

                        # Increment player score and reset if 10 points are reached
                        if player1_score == 10 or player2_score == 10:
                            winner_text = f'Player {winner} Wins!'
                            cv2.putText(img, text=winner_text, org=(int(img.shape[1] / 3), 200),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0),
                                        thickness=3)
                            cv2.imshow('Game', img)
                            player1_score = 0
                            player2_score = 0
                            score_pause_time = time.time() + 5  # Pause the score for 5 second

                cv2.putText(img, text=text, org=(int(img.shape[1] / 4), 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(148, 0, 211), thickness=2)

    if detect_hands and mode == 2:  # 모드가 2인 경우에만 실행
        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])

                if idx == 0 or idx == 6:  # fist or six
                    thumb_end = res.landmark[4]
                    fist_end = res.landmark[17]

                    text = None

                    if thumb_end.x - fist_end.x > THRESHOLD:
                        text = 'RIGHT'
                    elif fist_end.x - thumb_end.x > THRESHOLD:
                        text = 'LEFT'
                    elif thumb_end.y - fist_end.y > THRESHOLD:
                        text = 'DOWN'
                    elif fist_end.y - thumb_end.y > THRESHOLD:
                        text = 'UP'

                    if text is not None:
                        cv2.putText(img, text=text,
                                    org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                elif idx in [1, 2, 3, 4, 5, 9]:  # 숫자 1,2,3,4,5 인식
                    if idx == 9:
                        idx = 2

                    cv2.putText(img, text=str(idx),
                                org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    if mode == 0:  # 모드가 0인 경우에만 실행
        # Display "Start" text in the center of the screen
        start_text = "Start"
        text_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        cv2.putText(img, start_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    cv2.imshow('Game', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('a'):  # 'a' 키를 눌렀을 때 모드를 1로 변경
        mode = 1
    elif key == ord('b'):  # 'b' 키를 눌렀을 때 모드를 2로 변경
        mode = 2

cap.release()
cv2.destroyAllWindows()

