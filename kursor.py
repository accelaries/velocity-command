import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# ID ujung jari
finger_tips = [4, 8, 12, 16, 20]

# Ukuran layar
screen_w, screen_h = pyautogui.size()


# Fungsi hitung jari terbuka
def count_fingers(hand_landmarks):
    fingers = []

    # Thumb pakai orientasi X
    if (
        hand_landmarks.landmark[finger_tips[0]].x
        < hand_landmarks.landmark[finger_tips[0] - 1].x
    ):
        fingers.append(1)
    else:
        fingers.append(0)

    # Jari telunjuk sampai kelingking
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            total_fingers = count_fingers(hand_landmark)

            # Tampilkan angka jari terbuka
            cv2.putText(
                frame,
                str(total_fingers),
                (width - 50, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            # Jika 1 jari terbuka → gerak kursor
            if total_fingers == 1:
                x = int(hand_landmark.landmark[8].x * width)
                y = int(hand_landmark.landmark[8].y * height)

                # Konversi ke layar
                screen_x = np.interp(x, [0, width], [0, screen_w])
                screen_y = np.interp(y, [0, height], [0, screen_h])

                pyautogui.moveTo(screen_x, screen_y, duration=0.05)

                # Deteksi "tekan" dengan hitung jarak ibu jari & telunjuk
                thumb_tip = hand_landmark.landmark[4]
                index_tip = hand_landmark.landmark[8]
                distance = math.hypot(
                    index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y
                )

                if distance < 0.03:  # jika jari saling mendekat
                    pyautogui.click()
                    time.sleep(0.3)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
