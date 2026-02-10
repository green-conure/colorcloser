import cv2 as cv
import numpy as np
import time

capture = cv.VideoCapture(0)

cv.namedWindow("frame")

mouse_x, mouse_y = 0, 0


def get_mouse_pos(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


cv.setMouseCallback("frame", get_mouse_pos)

threshold = 50
color_detected_time = None


d_lowh, d_lows, d_lowv = 179, 255, 255
d_highh, d_highs, d_highv = 179, 255, 255

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_bound = np.array([d_lowh, d_lows, d_lowv])
    upper_bound = np.array([d_highh, d_highs, d_highv])

    mask = cv.inRange(hsv_frame, lower_bound, upper_bound)
    pixel_count = cv.countNonZero(mask)

    current_time = time.time()


    cv.putText(frame, "Press 'p' to pick color | 'q' to quit", (10, frame.shape[0] - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if pixel_count > threshold:
        if color_detected_time is None:
            color_detected_time = current_time
        else:
            elapsed_time = current_time - color_detected_time

            cv.putText(frame, f"DETECTING: {elapsed_time:.1f}s", (10, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if elapsed_time >= 1:

                cv.putText(frame, "Closing....", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv.imshow("frame", frame)
                cv.waitKey(1000)
                print("Color consistently detected for 1 second!")
                break
    else:
        color_detected_time = None

        #cv.putText(frame, "Scanning...", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv.imshow("frame", frame)
    #cv.imshow("mask", mask)

    key = cv.waitKey(1) & 0xFF


    if key == ord('p'):
        frozen_frame = frame.copy()
        hsv_array = []
        count = 0

        while count < 4:
            display_frame = frozen_frame.copy()


            cv.putText(display_frame, "Pick Color", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv.putText(display_frame, f"Sample {count}/4: Point & Press SPACE", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0), 2)

            cv.circle(display_frame, (mouse_x, mouse_y), 5, (255, 255, 255), 1)

            cv.imshow("frame", display_frame)

            key = cv.waitKey(1) & 0xFF

            if key == ord(' '):
                hsv_pixel = hsv_frame[mouse_y, mouse_x]
                hsv_array.append(hsv_pixel.tolist())


                cv.circle(frozen_frame, (mouse_x, mouse_y), 5, (0, 0, 255), -1)
                count += 1


        h_vals = [p[0] for p in hsv_array]
        s_vals = [p[1] for p in hsv_array]
        v_vals = [p[2] for p in hsv_array]

        padding = 10

        d_lowh = max(0, min(h_vals) - padding)
        d_lows = max(0, min(s_vals) - padding)
        d_lowv = max(0, min(v_vals) - padding)

        d_highh = min(179, max(h_vals) + padding)
        d_highs = min(255, max(s_vals) + padding)
        d_highv = min(255, max(v_vals) + padding)


        cv.putText(frozen_frame, "Yayyyyyy", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv.imshow("frame", frozen_frame)
        cv.waitKey(1000)

    if key == ord('q'):
        break

capture.release()
cv.destroyAllWindows()