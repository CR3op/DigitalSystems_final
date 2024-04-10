import cv2
import numpy as np

#Load the image
image_path = "Final_project/lung_ct.jpg"

try:
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load the image from ", image_path)
        exit()
    else:
        print("Image is loaded sucessfully")
except Exception as e:
    print("An error has happened: ", e)
    exit()
    
cv2.imshow("Image", img)
cv2.waitKey(0) #Wait for the key press to close the window
cv2.destroyAllWindows()

