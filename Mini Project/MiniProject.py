import cv2              # OpenCV 라이브러리를 'cv2'이름으로 가져옴
import mediapipe as mp  # Mediapipe 라이브러리를 'mp' 이름으로 가져옴
import numpy as np      # Numpy 라이브러리를 'mp' 이름으로 가져옴
import time             # 시간 관련 함수를 사용하기 위해 'time'모듈을 가져옴

max_num_hands = 2  # 최대 손 감지 수를 2로 설정
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}  # 가위바위보의 손 모양을 숫자와 문자열의 매핑으로 나타내는 딕셔너리를 생성

# MediaPipe hands model
mp_hands = mp.solutions.hands  # Mediapipe의 손 감지 모델을 사용
mp_drawing = mp.solutions.drawing_utils  # 손의 감지 결과를 시각적으로 표시하기 위한 유틸리티를 사용
hands = mp_hands.Hands(           # 손 감지 모델을 초기화
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')  # CSV 파일에서 제스처 학습 데이터를 가져옴
angle = file[:, :-1].astype(np.float32)  # 각도 데이터 가져옴
label = file[:, -1].astype(np.float32)  # 라벨 데이터 가져옴
knn = cv2.ml.KNearest_create()  # KNN분류기 생성
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # KNN분류기를 훈련

cap = cv2.VideoCapture(0)  # 비디오 캡처 장치 열기
cap.set(3, 640)  # 가로 해상도를 640으로 설정
cap.set(4, 480)  # 세로 해상도를 480으로 설정

player1_score = 0  # 플레이어 1의 점수를 초기화
player2_score = 0  # 플레이어 2의 점수를 초기화
detect_hands = True  # 손을 감지할 것인지 여부를 설정
score_pause_time = 0  # 점수를 표시하는 시간을 설정

while cap.isOpened():  # 비디오가 열려 있는 동안 반복

    ret, img = cap.read()  # 프레임을 읽기
    if not ret:
        continue

    img = cv2.flip(img, 1)  # 이미지를 좌우 반전(0 = 상하 반전/1 = 좌우 반전)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #  BGR에서 RGB로 변환

    if detect_hands:  # 손 감지가 활성화되어 있는 경우에만 실행
        result = hands.process(img)  # 손 감지 모델을 통해 이미지를 처리

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 이미지를 다시 BGR로 변환

        # Display player scores
        cv2.putText(img, text=f'Player 1: {player1_score}', org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)  # 플레이어 1의 점수를 화면에 표시
        cv2.putText(img, text=f'Player 2: {player2_score}', org=(img.shape[1] - 220, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)  # 플레이어 2의 점수를 화면에 표시

        if result.multi_hand_landmarks is not None:  # 손이 감지된 경우에 실행
            rps_result = []  # 손의 제스처를 저장할 빈 리스트를 초기화

            for res in result.multi_hand_landmarks:  # 'result.multi_hand_landmarks'에 포함된 각 손의 랜드마크에 대해 반복/'res'는 현재 처리 중인 손 나타냄
            rps_result = []

            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))  # 21x3 크기의 0으로 초기화된 배열을 만듬/21은 손의 각 랜드마크(지점)의 수, 3은 각 랜드마크의 x, y, z 좌표를 나타냄
                for j, lm in enumerate(res.landmark):  # 'res.landmark'에 포함된 각 랜드마크에 대해 반복/'j'는 현재 랜드마크의 인덱스, 'lm'은 현재 랜드마크를 나타냄
            rps_result = []

            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]  # 'joint[j] = [lm.x, lm.y, lm.z]': 현재 랜드마크의 x, y, z 좌표(이후 각도 게산에 이용)를 'joint' 배열에 저장
                    joint[j] = [lm.x, lm.y, lm.z]
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):  # enumerate() 함수는 순회 가능한(iterable) 객체(리스트, *튜플, 문자열 등)를 입력으로 받아 인덱스와 값을 순회 가능한 객체로 반환
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
                    org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))  # 'ord()' 함수는 문자의 유니코드 코드 포인트를 나타내는 정수를 반환함. 즉 문자열에서 주어진 문자의 유니코드 값을 반환함. 이 함수는 문자열에 포함된 문자 하나를 인자로 받음. 'chr()' 함수는 반대임.
                    cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    rps_result.append({
                        'rps': rps_gesture[idx],
                        'org': org
                    })

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # Who wins?
            if len(rps_result) >= 2:  # 제스처 결과가 두 개 이상인 경우에 실행
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


    cv2.imshow('Game', img)

    if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 프로그램을 종료
        break

cap.release()  # 비디오 캡처를 해제
cv2.destroyAllWindows()  # 모든 창을 닫기

