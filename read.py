import cv2 as cv


# Reading videos
capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier("haar_face.xml")
eye_cascade = cv.CascadeClassifier("haar_eye.xml")

while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray,1.3, 3)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
        cv.putText(frame, "Face", (x + 75, y), fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),thickness=1 )

    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), thickness=1)
        cv.putText(frame, "Eyes", (ex , ey), fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),thickness=1 )


    cv.imshow("Window", frame)
    if cv.waitKey(1) & 0xFF == ord('d'):
        print("Breaking...")
        break

capture.release()
cv.destroyAllWindows()
