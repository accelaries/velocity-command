import cv2
import mediapipe as mp
import subprocess  # Untuk membuka aplikasi
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

finger_tips = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks):
    fingers = []

    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# Flag agar Edge hanya dibuka sekali
edge_opened = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            total_fingers = count_fingers(hand_landmark)

            height, width, _ = frame.shape
            cv2.putText(
                frame, str(total_fingers),
                (width - 50, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

            # Jika 3 jari terbuka dan belum membuka Edge
            if total_fingers == 3 and not edge_opened:
                try:
                    subprocess.Popen("C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")  # Bisa juga: "start msedge", atau path lengkapnya
                    edge_opened = True
                    print("Edge dibuka!")
                except Exception as e:
                    print("Gagal membuka Edge:", e)

            # Reset kondisi jika jari turun jadi bukan 3
            elif total_fingers != 3:
                edge_opened = False

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
