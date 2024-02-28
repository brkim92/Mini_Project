import cv2                      # OpenCV 라이브러리를 'cv2'이름으로 가져옴
import mediapipe as mp          # Mediapipe 라이브러리를 'mp' 이름으로 가져옴
import numpy as np              # Numpy 라이브러리를 'mp' 이름으로 가져옴
import time                     # 시간 관련 함수를 사용하기 위해 'time'모듈을 가져옴
import tkinter as tk            # Python에서 GUI (Graphical User Interface)를 만들기 위한 표준 라이브러리
from tkinter import messagebox  # 사용자에게 메시지를 표시하는 데 사용/다양한 유형의 메시지 상자를 표시 가능

max_num_hands = 2               # 최대 손 감지 수를 2로 설정
THRESHOLD = 0.2  # 20%, 임계값이 낮을수록 손이 카메라와 가까워야만 제스처가 인식되고, 임계값이 높을수록 손이 카메라와 더 멀어져도 인식(임계값을 조정하여 원하는 거리에서 손의 제스처를 정확하게 인식할 수 있도록 할 수 있음)
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}  # 가위바위보의 손 모양을 숫자와 문자열의 매핑으로 나타내는 딕셔너리를 생성
gesture = {0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',}

mp_hands = mp.solutions.hands    # Mediapipe 라이브러리에서 제공하는 손 감지 모델을 사용하기 위해 'mp.solutions.hands' 모듈을 'mp_hands' 변수에 할당
mp_drawing = mp.solutions.drawing_utils  # 손의 감지 결과를 시각적으로 표시하기 위해 사용되는 유틸리티 함수를 포함하는 'mp.solutions.drawing_utils' 모듈을 'mp_drawing' 변수에 할당/손의 위치, 키포인트 및 연결된 선을 이미지에 그리는 데 사용
hands = mp_hands.Hands(            # 손 감지 모델을 초기화
    max_num_hands=max_num_hands,   # max_num_hands 매개변수는 감지해야 하는 최대 손의 수를 결정/최대 손 감지 수=2 설정
    min_detection_confidence=0.5,  # 손 감지의 최소 신뢰도 임계값
    min_tracking_confidence=0.5)   # 추적의 최소 신뢰도 임계값
    
mp_face_detection = mp.solutions.face_detection    # Mediapipe 라이브러리에서 얼굴 감지 모델을 사용하기 위해 'mp.solutions.face_detection' 모듈을 'mp_face_detection' 변수에 할당
face_detection = mp_face_detection.FaceDetection(  # 'mp_face_detection.FaceDetection' 클래스를 사용하여 얼굴 감지 객체를 생성
    min_detection_confidence=0.5)                  # 'min_detection_confidence' 매개변수는 얼굴을 감지하는 데 필요한 최소 신뢰도 임계값을 설정/감지된 얼굴에 대한 신뢰도<임계값(해당 얼굴 무시)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')  # NumPy의 'genfromtxt()'함수를 사용하여 CSV 파일에서 제스처 학습 데이터를 가져옴
angle = file[:, :-1].astype(np.float32)  # 각도 데이터 가져옴
label = file[:, -1].astype(np.float32)  # 라벨 데이터 가져옴
knn = cv2.ml.KNearest_create()  # KNN분류기 생성
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # KNN분류기를 훈련    

# 플레이어 1과 플레이어 2의 경계 상자를 나타내는 변수를 초기화/None은 경계 상자가 아직 지정되지 않았음을 나타냄
player1_bbox = None
player2_bbox = None
# 플레이어 1과 플레이어 2의 점수를 나타내는 변수를 초기화(초기값=0)
player1_score = 0
player2_score = 0
score_pause_time = 0  # 점수를 일시적으로 멈추는 데 사용되는 변수를 초기화(일시적인 중복 점수를 방지하기 위해 사용)

def save_player_images(player1_face, player2_face, cap):  # 플레이어 1과 플레이어 2의 얼굴 이미지를 저장하는 함수(매개변수 : player1_face, player2_face=얼굴 이미지, cap=이미지 저장에 사용되는 웹캠 객체)
    cv2.imwrite('player1.jpg', cv2.cvtColor(player1_face, cv2.COLOR_RGB2BGR))  # 함수는 얼굴 이미지를 JPEG 파일로 저장
    cv2.imwrite('player2.jpg', cv2.cvtColor(player2_face, cv2.COLOR_RGB2BGR))  # 함수는 얼굴 이미지를 JPEG 파일로 저장
    messagebox.showinfo("Image Saved", "Player images saved successfully!")  # 저장이 완료되면 메시지 창을 표시

    # 웹캠 종료
    cap.release()
    cv2.destroyAllWindows()

def detect_players(img, cap):
    global player1_score, player2_score, score_pause_time, player1_bbox, player2_bbox  # 전역 변수임을 명시

    results = face_detection.process(img)  # 입력 이미지에서 얼굴을 감지/'face_detection' 객체를 사용하여 이미지에서 얼굴을 찾기
    if results.detections:  # 이미지에서 얼굴이 감지된 경우에만 아래의 코드를 실행
        for i, detection in enumerate(results.detections):  # 각 얼굴 감지 결과에 대해 반복하면서 인덱스와 해당 감지 결과를 순회
            bboxC = detection.location_data.relative_bounding_box  # 얼굴 감지 결과에서 상대적인 경계 상자 정보를 가져옴
            ih, iw, _ = img.shape  # 입력 이미지의 높이와 너비를 가져옴
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)  # 상대적인 경계 상자 정보를 이미지 크기에 맞게 변환하여 절대적인 좌표로 변환

            # 감지된 얼굴의 인덱스에 따라 플레이어 1과 플레이어 2의 경계 상자를 설정
            player1_bbox = bbox if i == 0 else player1_bbox
            player2_bbox = bbox if i == 1 else player2_bbox

            # Draw rectangles around players
            cv2.rectangle(img, (player1_bbox[0], player1_bbox[1]), (player1_bbox[0]+player1_bbox[2], player1_bbox[1]+player1_bbox[3]), (0, 255, 0), 2)  # 이미지에 플레이어 1의 얼굴 주위에 초록색 사각형을 그림
            if player2_bbox:  # 플레이어 2의 경계 상자가 있는 경우에만 아래의 코드를 실행
                cv2.rectangle(img, (player2_bbox[0], player2_bbox[1]), (player2_bbox[0]+player2_bbox[2], player2_bbox[1]+player2_bbox[3]), (0, 255, 255), 2)  # 이미지에 플레이어 2의 얼굴 주위에 하늘색 사각형을 그림

                # Display Player 1 or Player 2 text based on face index
                player_text_color = (0, 0, 255) if i == 0 else (255, 0, 0)  # 플레이어 텍스트의 색상을 지정(첫 번째 플레이어는 빨간색, 두 번째 플레이어는 파란색으로 지정)
                player_text = f'Player {i + 1}'
                org = (int((player1_bbox[0] + player2_bbox[0]) / 2), int((player1_bbox[1] + player2_bbox[1]) / 2))  # 두 플레이어의 중심 위치를 계산
                cv2.putText(img, text=player_text, org=org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=player_text_color, thickness=2)  # 이미지의 중심 위치에 플레이어 번호를 표시, 색상은 전에 설정한 색상으로 지정

                # Save images
                player1_face = img[player1_bbox[1]:player1_bbox[1]+player1_bbox[3], player1_bbox[0]:player1_bbox[0]+player1_bbox[2]]  # 플레이어 1의 얼굴 이미지를 잘라냄
                player2_face = img[player2_bbox[1]:player2_bbox[1]+player2_bbox[3], player2_bbox[0]:player2_bbox[0]+player2_bbox[2]]  # 플레이어 2의 얼굴 이미지를 잘라냄
                save_player_images(player1_face, player2_face, cap)  # 잘라낸 플레이어 1과 플레이어 2의 얼굴 이미지를 저장하는 함수를 호출

    return img  # 처리된 이미지를 반환
    
def start_hand_detection():
    cap = cv2.VideoCapture(0)  # 비디오 캡처 장치 열기
    cap.set(3, 640)  # 가로 해상도를 640으로 설정
    cap.set(4, 480)  # 세로 해상도를 480으로 설정

    global player1_score, player2_score, score_pause_time, player1_bbox, player2_bbox
    player1_score = 0  # 플레이어 1의 점수를 초기화
    player2_score = 0  # 플레이어 2의 점수를 초기화
    score_pause_time = 0  # 점수를 표시하는 시간을 설정

    while cap.isOpened():  # 비디오가 열려 있는 동안 반복
        ret, img = cap.read()  # 프레임을 읽기
        if not ret:
            continue

        img = cv2.flip(img, 1)  # 이미지를 좌우 반전(0 = 상하 반전/1 = 좌우 반전)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR에서 RGB로 변환

        img = detect_hands(img)

        cv2.imshow('Game', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_hands(img):
    result = hands.process(img)  # 입력 이미지에서 손을 감지('hands' 객체를 사용하여 이미지에서 손을 찾기)
    global player1_score, player2_score, score_pause_time
    # Display player scores
    cv2.putText(img, text=f'Player 1: {player1_score}', org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)  # 플레이어 1의 점수를 화면에 표시(텍스트의 위치, 폰트, 크기, 색상)
    cv2.putText(img, text=f'Player 2: {player2_score}', org=(img.shape[1] - 220, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)  # 플레이어 2의 점수를 화면에 표시(이미지의 오른쪽 위에 표시)

    if result.multi_hand_landmarks is not None:  # 손이 감지된 경우에 아래의 코드를 실행
        rps_result = []  # 손의 제스처를 저장할 빈 리스트를 초기화

        for i, res in enumerate(result.multi_hand_landmarks):  # 감지된 각 손에 대해 반복하면서 손의 제스처를 분석('res'변수는 각 손의 랜드마크 정보 할당, 'i'변수는 해당 손의 인덱스 할당됨)
            joint = np.zeros((21, 3))  # 손의 각 관절 좌표를 저장할 배열을 생성
            for j, lm in enumerate(res.landmark):  # enumerate() 함수는 순회 가능한(iterable) 객체(리스트, *튜플, 문자열 등)를 입력으로 받아 인덱스와 값을 순회 가능한 객체로 반환
                joint[j] = [lm.x, lm.y, lm.z]  # 손의 각 관절 좌표를 저장

            # 각 관절 사이의 각도를 계산
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]  # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]  # Child joint
            v = v2 - v1  # [20,3] [손의 랜드마크 개수, 각 관절의 (x, y, z)좌표]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)  # 각 손의 제스처를 분석하기 위해 손의 각도 데이터를 NumPy 배열로 변환
            ret, results, neighbours, dist = knn.findNearest(data, 3)  # KNN (K-Nearest Neighbors) 알고리즘을 사용하여 손의 제스처를 분류(제스처에 대한 결과는 'results'에 저장)
            idx = int(results[0][0])  # 분류된 제스처 결과에서 가장 가능성이 높은 제스처의 인덱스를 가져옴

            # Draw gesture result
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))  # 'ord()' 함수는 문자의 유니코드 코드 포인트를 나타내는 정수를 반환함. 즉 문자열에서 주어진 문자의 유니코드 값을 반환함. 이 함수는 문자열에 포함된 문자 하나를 인자로 받음. 'chr()' 함수는 반대임.
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)  # 이미지에 인식된 제스처를 표시

                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

                # Display Player 1 or Player 2 text based on hand index(각 손의 제스처를 인식하여 플레이어 텍스트를 표시, 해당하는 플레이어의 색상을 지정)
                player_text_color = (0, 0, 255) if i == 0 else (255, 0, 0)
                player_text = f'Player {i + 1}'
                cv2.putText(img, text=player_text, org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=player_text_color, thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        # Who wins?
        if len(rps_result) >= 2:  # 인식된 제스처가 2개 이상인 경우에 아래의 코드를 실행하여 가위바위보 게임을 수행
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

            if winner is not None:  # 게임에서 승자가 결정된 경우에 아래의 코드를 실행
                cv2.putText(img, text='Winner', org=(int(img.shape[1] / 4), 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 255, 0), thickness=3)  # org=(int(img.shape[1] / 4), 70)는 이미지 내에 텍스트를 표시할 위치를 나타냄, org는 텍스트의 시작점을 나타내는 변수, (int(img.shape[1] / 4), 70)은 (x, y) 형태의 튜플
                if time.time() > score_pause_time:  # 현재 시간이 score_pause_time보다 클 경우에만 아래의 코드를 실행, 이 조건문은 일정 시간 동안 점수를 일시 정지하는 기능을 구현하기 위해 사용
                    player1_score += 1 if winner == 1 and time.time() > score_pause_time else 0  # 플레이어 1의 점수를 업데이트/winner == 1이 참이고 현재 시간이 score_pause_time보다 클 경우에만 플레이어 1의 점수를 1 증가/점수를 일시 정지 중인 경우 아무 작업도 수행하지 않음
                    player2_score += 1 if winner == 2 and time.time() > score_pause_time else 0 # 플레이어 2의 점수를 업데이트/winner == 2이 참이고 현재 시간이 score_pause_time보다 클 경우에만 플레이어 1의 점수를 1 증가/점수를 일시 정지 중인 경우 아무 작업도 수행하지 않음
                    score_pause_time = time.time() + 2  # 현재 시간에서 2초를 더하여 score_pause_time을 업데이트/점수를 2초 동안 일시 정지 이후에는 다시 점수가 업데이트됨

                    # Increment player score and reset if 10 points are reached
                    if player1_score == 10 or player2_score == 10:
                        winner_text = f'Player {winner} Wins!'
                        cv2.putText(img, text=winner_text, org=(int(img.shape[1] / 3), 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                        cv2.imshow('Game', img)
                        player1_score = 0
                        player2_score = 0
                        score_pause_time = time.time() + 5  # Pause the score for 5 second

            cv2.putText(img, text=text, org=(int(img.shape[1] / 4), 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(148, 0, 211), thickness=2)  # 이미지에 게임의 결과를 나타내는 텍스트를 표시

    return img  # 처리된 이미지를 반환
    
def gesture(img):  # 이미지에서 손의 제스처를 분석하는 함수를 정의            
    result = hands.process(img)  # 'hands 객체를 사용하여 입력 이미지에서 손을 감지
    if result.multi_hand_landmarks is not None:  # 손이 감지된 경우에 아래의 코드 블록을 실행
        for res in result.multi_hand_landmarks:  # 각 손에 대해 반복
            joint = np.zeros((21, 3))  # 손의 각 관절 좌표를 저장할 배열을 생성
            for j, lm in enumerate(res.landmark):  # 각 손의 관절 좌표를 배열에 저장
                joint[j] = [lm.x, lm.y, lm.z]

            # 각 관절 사이의 벡터를 계산
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20,3] [손의 랜드마크 개수, 각 관절의(x, y, z)좌표]
            # 벡터의 정규화
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # 관절 사이의 각도를 계산
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # 라디안에서 각도 다시 변환

            # 각도 데이터를 KNN 알고리즘을 사용하여 제스처로 분류
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])
            
            # 제스처가 주먹이거나 여섯이면 아래의 코드 블록을 실행
            if idx == 0 or idx == 6:
                # 주먹의 위치에 따라 방향을 결정하고, 이를 텍스트로 저장
                thumb_end = res.landmark[4]  # 주먹을 쥐었을 때의 엄지 손가락 끝의 위치를 나타냄
                fist_end = res.landmark[17]  # 주먹을 쥐었을 때의 손의 끝 부분(쥔 부분)의 위치를 나타냄

                text = None
                
                # 두 랜드마크 사이의 상대적인 위치를 비교하여 주먹의 방향을 결정합니다. 여기서 THRESHOLD 값은 주먹의 위치를 결정할 때 사용되는 임계값으로, 손의 카메라와의 거리에 따라 방향이 결정
                if thumb_end.x - fist_end.x > THRESHOLD:  # 'thumb_end.x - fist_end.x' 값 > THRESHOLD(주먹이 오른쪽으로 이동했다고 판단하여 텍스트를 'RIGHT'로 설정)
                    text = 'RIGHT'
                elif fist_end.x - thumb_end.x > THRESHOLD:  # 'fist_end.x - thumb_end.x' 값 > THRESHOLD(주먹이 왼쪽으로 이동했다고 판단하여 텍스트를 'LEFT'로 설정)
                    text = 'LEFT'
                elif thumb_end.y - fist_end.y > THRESHOLD:  # 'thumb_end.y - fist_end.y' 값 > THRESHOLD(주먹이 아래쪽으로 이동했다고 판단하여 텍스트를 'DOWN'로 설정)
                    text = 'DOWN'
                elif fist_end.y - thumb_end.y > THRESHOLD:  # 'fist_end.y - thumb_end.y' 값 > THRESHOLD(주먹이 위쪽으로 이동했다고 판단하여 텍스트를 'UP'로 설정)
                    text = 'UP'

                if text is not None:  # 결정된 방향을 이미지에 텍스트로 표시
                    cv2.putText(img, text=text,
                                org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            elif idx in [1, 2, 3, 4, 5, 9]:  # 제스처가 숫자 1, 2, 3, 4, 5 중 하나인 경우 아래의 코드 블록을 실행
                # 제스처가 9인 경우 2로 변환
                if idx == 9:
                    idx = 2

                # 인식된 숫자를 이미지에 텍스트로 표시
                cv2.putText(img, text=str(idx),
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)  # 감지된 손의 랜드마크를 이미지에 그림
    return img  # 처리된 이미지를 반환

def start_game(mode):  # mode 매개변수를 받는 함수를 정의(이 함수는 게임을 시작하고 감지 모드에 따라 이미지를 처리하고 화면에 표시)
    cap = cv2.VideoCapture(0)  # 웹캠을 통해 비디오를 캡처하기 위한 VideoCapture 객체를 생성

    global player1_score, player2_score, score_pause_time  # 플레이어 1 및 플레이어 2의 점수와 게임 점수를 일시 정지하는 데 사용되는 시간을 전역 변수로 초기화
    player1_score = 0
    player2_score = 0
    score_pause_time = 0

    while cap.isOpened():  # 웹캠이 열려있는 동안 무한 루프를 실행
        ret, img = cap.read()  # 웹캠에서 프레임을 읽어옴 
        if not ret:  # 프레임을 읽어오는 것에 실패한 경우 다음 프레임으로 넘어감
            continue

        img = cv2.flip(img, 1)  # 웹캠으로부터 읽어온 이미지를 좌우 반전(1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 색상 형식을 RGB 색상 형식으로 변환

        # mode에 따라 이미지를 처리
        if mode == 1:  # mode가 1이면 손 감지를 실행
            img = detect_hands(img)
        elif mode == 2:  # mode가 2이면 플레이어 감지를 실행
            img = detect_players(img, cap)  # cap 변수를 전달
        elif mode == 3:  # mode가 3이면 제스처 감지를 실행
            img = gesture(img) 

        cv2.imshow('Game', img)  # 처리된 이미지를 게임 창에 표시

        if cv2.waitKey(1) == ord('q'):  # 사용자가 'q' 키를 누르면 루프를 종료(=게임 종료)/종료 키를 누를 때까지 대기하다가 종료 키가 눌리면 루프 탈출
            break

    cap.release()  # 비디오 캡처를 해제
    cv2.destroyAllWindows()  # 모든 창을 닫기

# GUI 생성
def create_start_window():  # 시작 창을 생성하는 함수를 정의
    start_window = tk.Tk()  # Tkinter에서 'TK()'함수를 사용하여 시작 창을 생성
    start_window.title("mini kimbap")  # 'title()'로 창의 제목을 "mini kimbap"으로 설정

    # Face Detection 버튼
    face_button = tk.Button(start_window, text="플레이어 등록", command=lambda: start_game(mode=2))  # 플레이어 등록을 위한 버튼을 생성/해당 버튼을 누를 경우 start_game 함수를 호출하여 플레이어 감지 모드(2)로 게임을 시작
    face_button.pack(pady=10)  # 'pack()'를 사용하여 버튼을 창에 배치

    # Hand Detection 버튼
    hand_button = tk.Button(start_window, text="Game Mode", command=lambda: start_game(mode=1))  # 게임 모드를 위한 버튼을 생성/해당 버튼을 누를 경우 start_game 함수를 호출하여 손 감지 모드(1)로 게임을 시작
    hand_button.pack(pady=10)  # 'pack()'를 사용하여 버튼을 창에 배치
    
    # Gesture Detection 버튼
    hand_button = tk.Button(start_window, text="Gesture Mode", command=lambda: start_game(mode=3))  # 제스처 감지 모드를 위한 버튼을 생성/해당 버튼을 누를 경우 start_game 함수를 호출, 제스처 감지 모드(3)로 게임을 시작
    hand_button.pack(pady=10)  # 'pack()'를 사용하여 버튼을 창에 배치

    start_window.mainloop()  # 'mainloop()'를 사용하여 창을 실행/시작 창을 루프하며 유지/사용자가 창을 닫을 때까지 이벤트를 처리

# 시작 창 열기(사용자가 버튼을 누르면 start_game 함수가 호출되어 각 모드에 따른 게임이 시작)
create_start_window()
