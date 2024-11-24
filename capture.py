import cv2
from datetime import datetime

first_frame = None
status_list = []  # Initialize an empty list for status updates
status2 = 0
times = []
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    check, frame = video.read()
    status = 0

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (21, 21), 0)

    if first_frame is None:
        first_frame = grey
        continue

    delta_frame = cv2.absdiff(first_frame, grey)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # In newer versions of OpenCV, findContours returns only two values
    _, cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    if status != status2:
        times.append(datetime.now())
        status2 = status

    cv2.imshow("Grey frame", grey)
    cv2.imshow("Delta frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)  # Corrected typo in window name
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(status_list)  # Remove unnecessary print statement
print(times)

video.release()
cv2.destroyAllWindows()
