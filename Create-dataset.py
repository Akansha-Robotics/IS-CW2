# OpenCV library 
import cv2 as cv
# Connection to Operating system
import os

# Verify images folder exists to store captured images 
# If image folder does not exist
if not os.path.exists('images'):
    # Create the image folder 
    os.makedirs('images')

# Function to capture image 
def capture_image():
    # Initialize webcam and set to default which is 0 
    cap = cv.VideoCapture(0)
    # Prints prompts that use can use 
    print("Space: To Capture Image; ESC: To Exit")

    # When the loop is running 
    while True:
        # Capture the frame from webcam and is held in frame variable 
	    # ret means boolean to indicate whether frame is captured or not
        ret, frame = cap.read()

        # Display video frame being captured and sets title of window
        cv.imshow('Captured Frame', frame)

        # Awaiting for key press to be saved in key variable 
        # cv2.waitKey converts the key into ASCII so it can be understood by python and used in if loop
        key = cv.waitKey(1)

        # If loop for space being pressed 
        # key % 256 == 32 indicates the ASCII for space button pressed 
        if key % 256 == 32:
            # Request user input for name of file to be saved - name stored in variable
            card_name = input("Name For Image: ")
            # Sets the card name as user inputted for saving in folder by communicating with operating system (os.path.join)
            # Sets the folder to be  images folder and png file type 
            image_path = os.path.join('images', f'{card_name}.png')
            # Saves image in folder accordingly to image_path variable set
            cv.imwrite(image_path, frame)
            # Prints confirmation message of image saved in path 
            print(f'Image Saved As {image_path}')

        # If loop for Esc being pressed 
        # key % 256 == 27 indicates the ASCII for Esc button pressed
        elif key % 256 == 27:
            # Break the loop and exits 
            break

    # As the while loop is ended, release then webcam 
    cap.release()
    # and close all the windows 
    cv.destroyAllWindows()

# Calls the function for capturing image 
capture_image()