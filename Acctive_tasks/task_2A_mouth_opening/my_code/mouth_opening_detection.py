import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt

# img = cv2.imread('./face_model (1).jpg')
# img = cv2.resize(img, (1000,1000))

video = cv2.VideoCapture(0)
while True:
    
    ret, frame = video.read()
   
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        x1,y1 = face.left(), face.top()
        x2,y2 = face.right(), face.bottom()
        imgOriginal = cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),2)
        landmarks = predictor(imgGray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # cv2.circle(imgOriginal, (x,y), 5, (50,50,255), cv2.FILLED)
            if (n == 52):
                x_1, y_1 = landmarks.part(n).x, landmarks.part(n).y

            if (n == 57):
                x_2, y_2 = landmarks.part(n).x, landmarks.part(n).y
            
            if (n == 48):
                x_3, y_3 = landmarks.part(n).x, landmarks.part(n).y
            
            if (n == 54):
                x_4, y_4 = landmarks.part(n).x, landmarks.part(n).y

            # if (n>=48 and n<=60): # (48, 54), (51, 57)
            #     cv2.putText(imgOriginal, str(n), (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1) # lip (48, 60)
                # print(f"number {n} has (x:{x}, y:{y})")
            
        final_img = cv2.rectangle(imgOriginal, (x_3, y_1), (x_4, y_2),(100,100,0),2 )

    # Our operations on the frame come here

    # Display the resulting frame
    cv2.imshow('frame',final_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()        
cv2.destroyAllWindows()