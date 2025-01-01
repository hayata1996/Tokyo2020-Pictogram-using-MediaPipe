# this is the simplest code to capture video from camera
# and display
# To stop recording, press 'q'

import cv2
import time

# カメラからビデオをキャプチャ
cap = cv2.VideoCapture(0)

# FPS計算のための初期化
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 現在の時間を取得
    curr_time = time.time()
    # FPSを計算
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # フレームにFPSを表示
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    # 0.1秒待つ
    # time.sleep(0.1)
    # time.sleepは秒単位、waitkeyはミリ秒単位
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
