import cv2
# to enable camera of the desktop
# video_capture = cv2.VideoCapture(0)
# while True:
#     ret, frame = video_capture.read()
#     cv2.imshow('Video', frame) 
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break    
# video_capture.release()


face_capture = cv2.CascadeClassifier("C:/Users/KIIT0001/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_capture.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30), 
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Video', frame) 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break    
video_capture.release()