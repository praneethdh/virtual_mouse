import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# ───────────── SETUP ─────────────
wCam, hCam = 640, 480
frameR = 100
smoothening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

last_left_click = 0
last_right_click = 0
last_double_click = 0
scroll_anchor_y = None
prev_fingers = [0, 0, 0, 0, 0]
pinch_active = False
pinch_start_pos = None
moved_during_pinch = False
action = ""
mouse_mode = False
ok_gesture_last_state = False

def fingers_up(lmList):
    tips = [8, 12, 16, 20]
    up = []
    if lmList[4][1] > lmList[3][1]:
        up.append(1)
    else:
        up.append(0)
    for tip in tips:
        if lmList[tip][2] < lmList[tip - 2][2]:
            up.append(1)
        else:
            up.append(0)
    return up

def find_distance(p1, p2, lmList):
    x1, y1 = lmList[p1][1], lmList[p1][2]
    x2, y2 = lmList[p2][1], lmList[p2][2]
    return math.hypot(x2 - x1, y2 - y1)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        lmList = []
        for id, lm in enumerate(handLms.landmark):
            cx, cy = int(lm.x * wCam), int(lm.y * hCam)
            lmList.append([id, cx, cy])

        if lmList:
            x_index, y_index = lmList[8][1], lmList[8][2]
            x_middle, y_middle = lmList[12][1], lmList[12][2]
            x_ring, y_ring = lmList[16][1], lmList[16][2]

            curr_fingers = fingers_up(lmList)
            dist_thumb_index = find_distance(4, 8, lmList)
            is_ok_gesture = dist_thumb_index < 30 and curr_fingers[2] == 1 and curr_fingers[3] == 1 and curr_fingers[4] == 1

            # Toggle mouse mode on OK gesture transition (rising edge)
            if is_ok_gesture and not ok_gesture_last_state:
                mouse_mode = not mouse_mode
                action = "Mouse Mode ON" if mouse_mode else "Mouse Mode OFF"
                time.sleep(0.5)  # debounce
            ok_gesture_last_state = is_ok_gesture

            if mouse_mode:
                dist_im = find_distance(8, 12, lmList)

                if curr_fingers[1] == 1 and curr_fingers[2] == 1 and dist_im < 30:
                    if not pinch_active:
                        pinch_active = True
                        pinch_start_pos = pyautogui.position()
                        moved_during_pinch = False
                        pyautogui.mouseDown(button="left")
                        action = "Pinch Grab (Hold)"
                        scroll_anchor_y = None
                    else:
                        action = "Pinch Grab (Hold)"
                    x_screen = np.interp(x_index, (frameR, wCam - frameR), (0, screen_width))
                    y_screen = np.interp(y_index, (frameR, hCam - frameR), (0, screen_height))
                    clocX = plocX + (x_screen - plocX) / smoothening
                    clocY = plocY + (y_screen - plocY) / smoothening
                    pyautogui.moveTo(clocX, clocY)
                    cur_mouse_pos = pyautogui.position()
                    if not moved_during_pinch:
                        dx = abs(cur_mouse_pos.x - pinch_start_pos.x)
                        dy = abs(cur_mouse_pos.y - pinch_start_pos.y)
                        if dx > 5 or dy > 5:
                            moved_during_pinch = True
                    plocX, plocY = clocX, clocY
                    prev_fingers = curr_fingers.copy()
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (0, 0), (350, 40), (0, 0, 0), -1)
                    cv2.putText(img, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Virtual Mouse", img)
                    if cv2.waitKey(1) == ord('q'):
                        break
                    continue

                if pinch_active and (curr_fingers[1] == 0 or curr_fingers[2] == 0 or dist_im >= 30):
                    pyautogui.mouseUp(button="left")
                    pinch_active = False
                    if not moved_during_pinch:
                        pyautogui.doubleClick()
                        action = "Pinch Release → DoubleClick"
                        last_double_click = time.time()
                    else:
                        action = "Release Grab"
                    scroll_anchor_y = None

                if not pinch_active:
                    if curr_fingers[0] == 0:
                        if curr_fingers[1] == 1 and curr_fingers[2] == 0 and curr_fingers[3] == 0:
                            x_screen = np.interp(x_index, (frameR, wCam - frameR), (0, screen_width))
                            y_screen = np.interp(y_index, (frameR, hCam - frameR), (0, screen_height))
                            clocX = plocX + (x_screen - plocX) / smoothening
                            clocY = plocY + (y_screen - plocY) / smoothening
                            pyautogui.moveTo(clocX, clocY)
                            plocX, plocY = clocX, clocY
                            action = "Move"
                        else:
                            action = ""
                            scroll_anchor_y = None
                    else:
                        if (curr_fingers[1] == 1 and curr_fingers[2] == 1 and curr_fingers[3] == 1):
                            center_y = (y_index + y_middle + y_ring) // 3
                            if scroll_anchor_y is None:
                                scroll_anchor_y = center_y
                                action = "Scroll"
                            else:
                                dy = scroll_anchor_y - center_y
                                if abs(dy) > 5:
                                    scroll_amt = int(dy * 2)
                                    pyautogui.scroll(scroll_amt)
                                    action = "Scroll"
                        elif (prev_fingers[1] == 1 and prev_fingers[2] == 1 and curr_fingers[1] == 0 and curr_fingers[2] == 1):
                            if time.time() - last_left_click > 1:
                                pyautogui.click(button="left")
                                action = "Left Click"
                                last_left_click = time.time()
                            scroll_anchor_y = None
                        elif (prev_fingers[1] == 1 and prev_fingers[2] == 1 and curr_fingers[1] == 1 and curr_fingers[2] == 0):
                            if time.time() - last_right_click > 1:
                                pyautogui.click(button="right")
                                action = "Right Click"
                                last_right_click = time.time()
                            scroll_anchor_y = None
                        elif (curr_fingers[1] == 1 and curr_fingers[2] == 0 and curr_fingers[3] == 0):
                            action = "Frozen"
                            scroll_anchor_y = None
                        else:
                            action = ""
                            scroll_anchor_y = None
                prev_fingers = curr_fingers.copy()
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if action:
        cv2.rectangle(img, (0, 0), (350, 40), (0, 0, 0), -1)
        cv2.putText(img, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.rectangle(img, (0, 0), (350, 40), (0, 0, 0), -1)
        cv2.putText(img, "Action: None", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
