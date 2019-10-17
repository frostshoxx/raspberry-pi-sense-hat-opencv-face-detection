import cv2
import random
from sense_hat import SenseHat

# initialize the app
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
sense = SenseHat()
cap = cv2.VideoCapture(0)
window_name = "Jaffar's Pi Bot: OpenCV Facial Detection"

# loop runs if capturing has been initialized. 
while 1: 

    # reads frames from a camera 
    ret, img = cap.read() 

    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)         
    
    for (x,y,w,h) in faces:
        
        # To draw a rectangle in a face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 

        # Detects eyes of different sizes in the input image 
        eyes = eye_cascade.detectMultiScale(roi_gray) 
            
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 

    # Display an image in a window 
    cv2.imshow(window_name,img)
    cv2.moveWindow(window_name, 1100, 250);
    
    # Display the number of detected faces
    if (len(faces) > 0):
        sense.show_letter(str(len(faces)),(0, random.randint(20,80), 0))
    else:
        for x in range(8):
            sense.set_pixel(x, random.randint(0,7), (0, random.randint(20,80), 0))  

    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

# Close the window 
cap.release()
sense.clear()

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 

