# OpenCV library 
import cv2 as cv
# NumPy library to do numerical operations such as calculating bounding box and other required calculations 
import numpy as np
# Connection to Operating system
import os

# Function to process frame and find ROI by either normal processing or using center rule logic 
def process_frame(frame):
    # If normal processing is true
    if not use_center_rule:
        # Starts from first converting to grayscale using cvtColor
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        # Thresholding - THRESH_BINARY_INV sets values accordingly to threshold value so that it is set to either 0 or 255
        _, binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)
    
        # Conduct morphological dilation - make numbers larger and close gaps 
        # MORPH_ELLIPSE is used for getting circular structuring element which is suited for numbers on UNO card 
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # Dilated image to better find contours 
        dilated = cv.morphologyEx(binary, cv.MORPH_DILATE, kernel)
    
        # Find contours - external ones (RETR_EXTERNAL) and edges of shape (CHAIN_APPROX_SIMPLE)
        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Calculates and finds the center of dilated image  
        center = np.array([dilated.shape[1]/2, dilated.shape[0]/2])
        # Initialize variable for closest contour found
        closest_contour = None
        # and to store min distance between center and contours 
        min_dist = np.inf

        # For loop to find good contour closest to center 
        for contour in contours:
            # Calculate elements for the contour like shape and region located 
            M = cv.moments(contour)
            # If contour is not 0 and greater than 20 
            if M['m00'] != 0 and cv.contourArea(contour) > 20:
                # Calculate the centroid x and y values of contour 
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # Calculate distance between the centroid and the frame center
                distance = np.linalg.norm(center - np.array([cx, cy]))
                # if the distance between centroid and the frame center is smaller than the minimum distance found from other contours 
                if distance < min_dist:
                    # Make that distance which is smaller as min distance 
                    min_dist = distance
                    # and make that contour the closest one 
                    closest_contour = contour

        # If closest contour is found 
        if closest_contour is not None:
            # Calculate bounding box coordinates
            x, y, w, h = cv.boundingRect(closest_contour)
            # Extend the height and width of box 
            vertical_extension = int(h * 0.25)
            horizontal_extension = int(w * 0.25)
            # Learning Point: Was facing issue where ROI was cutting the number so added this extensions to allow for whole number to be taken 

            # Applies extension value set above to bounding box 
            y = max(y - vertical_extension, 0)
            h = min(h + (2 * vertical_extension), frame.shape[0] - y)
            x = max(x - horizontal_extension, 0)
            w = min(w + (2 * horizontal_extension), frame.shape[1] - x)

            # Gets ROI from extended bounding box 
            roi = frame[y:y+h, x:x+w]

            # Applies function to further process ROI 
            final_roi = process_extracted_roi(roi)

            # ROI found and set to true 
            return final_roi, True

        # No ROI found and set to false 
        return None, False 
    
    # If center rule is set to true 
    else:
        # Starts from first converting to grayscale using cvtColor
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Thresholding - THRESH_BINARY_INV sets values accordingly to threshold value so that it is set to either 0 or 255
        _, binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)

        # Conduct morphological dilation - make numbers larger and close gaps
        # MORPH_ELLIPSE is used for getting circular structuring element which is suited for numbers on UNO card
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # Dilated image to better find contours
        dilated = cv.morphologyEx(binary, cv.MORPH_DILATE, kernel)

        # Find contours - external ones (RETR_EXTERNAL) and edges of shape (CHAIN_APPROX_SIMPLE)
        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Calculates and finds the center of frame 
        center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
        # Initialize variable for closest contour found
        closest_contour = None
        # and to store min distance between center and contours
        min_dist = np.inf

        # Loop through each contour 
        for contour in contours:
            # Calculate elements for the contour like shape and region located
            M = cv.moments(contour)
            # If contour is not 0 and greater than 20
            if M['m00'] != 0 and cv.contourArea(contour) > 20:
                # Calculate the centroid x and y values of contour
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Calculate distance between the centroid and the frame center
                distance = np.linalg.norm(center - np.array([cx, cy]))
                # if the distance between centroid and the frame center is smaller than the minimum distance found from other contours
                if distance < min_dist:
                    # Make that distance which is smaller as min distance
                    min_dist = distance
                    # and make that contour the closest one
                    closest_contour = contour

        # If closest contour is found
        if closest_contour is not None:
            # Calculate bounding box coordinates
            x, y, w, h = cv.boundingRect(closest_contour)

            # Extend the height and width of box
            vertical_extension = int(h * 0.25)
            horizontal_extension = int(w * 0.25)
            # Learning Point: Was facing issue where ROI was cutting the number so added this extensions to allow for whole number to be taken

            # Applies extension value set above to bounding box
            y = max(y - vertical_extension, 0)
            h = min(h + (2 * vertical_extension), frame.shape[0] - y)
            x = max(x - horizontal_extension, 0)
            w = min(w + (2 * horizontal_extension), frame.shape[1] - x)

            # Gets ROI from extended bounding box
            roi = frame[y:y+h, x:x+w]

            # Applies function to further process ROI
            final_roi = process_extracted_roi(roi)

            # ROI found and set to true
            return final_roi, True

        # No ROI found and set to false
        return None, False 

# Function of extracting ROI further 
def process_extracted_roi(roi):
    # Sets kernel sizing 
    kernel = np.ones((3, 3), np.uint8)
    # Conducts closing operation - clear out any small gaps and missing parts 
    closed_roi = cv.morphologyEx(roi, cv.MORPH_CLOSE, kernel)

    # Conduct erosion to clear further gaps 
    closed_roi = cv.erode(closed_roi, kernel, iterations=1)
    # Conduct dilation to increase size of number in ROI 
    closed_roi = cv.dilate(closed_roi, kernel, iterations=1)

    # Returns extracted ROI 
    return closed_roi

# Function to access images dataset and create templates according to number label
def load_templates(template_dir):
    # Creates dictionary to hold templates and number label 
    templates = {}
    # Loop through files in images folder 
    for filename in os.listdir(template_dir):
        # Checks if file is png format 
        if filename.endswith(".png"):
            # Takes the number label from file name - splits the name to capture only that element 
            number = filename.split('-')[1].split('.')[0]
            # Reads the image template as grayscale 
            template = cv.imread(os.path.join(template_dir, filename), cv.IMREAD_GRAYSCALE)
            # If template exists
            if template is not None:
                # Add template with number label as the key in dictionary
                templates[number] = template
    # Return final template dictionary 
    return templates

# Function matches the ROI extracted from webcam and templates 
def match_template(roi, templates):
    # Starts from first converting to grayscale using cvtColor
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    # For holding the number in which the ROI matches the template best 
    matched_number = None
    # For holding similarity score of the matched template and ROI 
    highest_score = 0

    # Loop through each template stored 
    for number, template in templates.items():
        # Conducts template matching 
        # TM_CCOEFF_NORMED indicates template matching technique from OpenCV Library 
        # using cross-correlation coefficient calculate similarity along with being normalized with NORMED
        result = cv.matchTemplate(gray_roi, template, cv.TM_CCOEFF_NORMED)
        # Find the highest similarity
        _, max_val, _, _ = cv.minMaxLoc(result)

        # If the max_value is greater than highest score 
        if max_val > highest_score:
            # It will update that value as the highest score 
            highest_score = max_val
            # and set that number label as the matched number 
            matched_number = number

    # Returns matched number label and highest score
    return matched_number, highest_score


# Indicate path where images data set is stored 
template_dir = 'C:/Users/bhati/Python Code/IS-CW2/images'
# Applies load_templates function to start processing the labeled dataset 
templates = load_templates(template_dir)

# Initialize capture of video stream 
cap = cv.VideoCapture(0)

# Sets the use_center_rule to off by default 
use_center_rule = False  

# Main loop for number recognition
while True:
    # Capture the frame from webcam and is held in frame variable 
	# ret means boolean to indicate whether frame is captured or not
    ret, frame = cap.read()
    # if ret is true and frame is captured 
    if ret:
        # Applies the process_frame function to frame 
        if use_center_rule:
            roi, roi_detected = process_frame(frame)
        else:
            roi, roi_detected = process_frame(frame)
        # use_center_rule application is decided in process_frame function 

        # If ROI is found 
        if roi_detected:
            # Display the extracted ROI
            cv.imshow('Extracted ROI', roi)

            # Applies the match_template function to get number prediction 
            matched_number, score = match_template(roi, templates)

            # If there is a matched number 
            if matched_number:
                # Draw green dot at ROI center 
                cv.circle(roi, (roi.shape[1] // 2, roi.shape[0] // 2), 5, (0, 255, 0), -1)

                # Writes prediction result on frame window 
                cv.putText(frame, f'Number: {matched_number} Score: {score:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display webcam stream
        cv.imshow('Frame', frame)

    # Check if key is press - wait for 1 millisecond
    key = cv.waitKey(1)

    # If q key is pressed 
    if key == ord('q'):
        # Breaks the loop and exit the program 
        break
    # If c key is pressed 
    elif key == ord('c'):
        # Activates the use_center_rule to be true 
        use_center_rule = not use_center_rule 
        # If use_center_rule is true
        if use_center_rule:
            # Prints message for user 
            print("Center rule activated.")
        # If use_center_rule is not true 
        else:
            # Prints message for user
            print("Center rule deactivated.")

# Release the webcam
cap.release()
# and close all the windows
cv.destroyAllWindows()



