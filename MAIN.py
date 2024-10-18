import cv2
import imutils

video_capture = cv2.VideoCapture(0)

first_frame = None
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if first_frame is None:
        first_frame = blurred_frame
        continue


    frame_delta = cv2.absdiff(first_frame, blurred_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frame_delta)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break



video_capture.release()
cv2.destroyAllWindows()
