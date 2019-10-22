import cv2
import random
from sense_hat import SenseHat

# initialize the app
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
sense = SenseHat()
cap = cv2.VideoCapture(0)
window_name = "Jaffar's Pi Bot: OpenCV Facial Detection"
font = cv2.FONT_HERSHEY_SIMPLEX  
fontScale = 1   
color = (0, 0, 80)   
thickness = 2

# loop runs if capturing has been initialized. 
while 1: 

    # reads frames from a camera 
    ret, img = cap.read()
    sense.clear()
    camera_frame = cv2.flip(img,1)
    width = cap.get(3)  # float
    height = cap.get(4) # float

    # convert to gray scale of each frames 
    gray = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY) 

    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)         
    
    for (x,y,w,h) in faces:        
        # To draw a rectangle in a face 
        cv2.rectangle(camera_frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        # Put some scouter data display
        camera_frame = cv2.putText(camera_frame, 'Power', (x+80,y-50), font, fontScale, color, thickness, cv2.LINE_AA)
        camera_frame = cv2.putText(camera_frame, str(random.randint(10000,65535)), (x+70,y-20), font, fontScale, color, thickness, cv2.LINE_AA)
        
        # Map the detected faces on Pi Sense Hat LED
        for px in range(int(round(7 * x / width)), int(round(7 * (x+w) / width))+1):
            for py in range(int(round(7 * y / height)),int(round(7 * (y+h) / height))+1):
                sense.set_pixel(px, py, color)

    # Display an image in a window 
    cv2.imshow(window_name,camera_frame)
    cv2.moveWindow(window_name, 900, 450); #control where the window should be on the monitor here
    
    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

# Close the window 
cap.release()
sense.clear()

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 

