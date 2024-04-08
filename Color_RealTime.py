# OpenCV library 
import cv2 as cv
# NumPy library to do numerical operations such as calculating bounding box and other required calculations 
import numpy as np
# Keras library from TensorFlow for image recognition 
from tensorflow import keras 
# Library to allow for pre-trained keras model to be used  
from keras.models import load_model
# Library for handling data insertion and buffer - Detection of changes in hue 
from collections import deque
# Library to do calculation of mode -  the most common value - Detection of dominate hue 
from statistics import mode

# Main function for real time processing of frame and ROI
def process_frame(frame):
    # Starts from first converting to grayscale using cvtColor
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Gaussian blur for reducing noise and lead to better result for finding ROI 
    # (5,5) is the kernel size set to determine how string blurring will be done 
    # 0 is for standard deviation which is set to default and calculated by OpenCV 
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to catch edges
    # 255 means pixel intensity value which is set to max intensity 
    # ADAPTIVE_THRESH_GAUSSIAN_C does calculation for weighted sum of neighborhood value while THRESH_BINARY_INV sets values accordingly to threshold value so that it is set to either 0 or 255 
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    
    # Kernel size set to (5,5)
    kernel = np.ones((5,5),np.uint8)
    # Dilated image to better find contours
    dilated = cv.dilate(thresh, kernel, iterations=1)
    # Learning Point: Faced issue where yellow and blue cards were not being recognized - Upon adding dilation, it immediately fixed the error and was able to recognize color 

    # Find contours - external ones (RETR_EXTERNAL) and edges of shape (CHAIN_APPROX_SIMPLE)
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by remove very larger or very small one 
    contours = sorted([cnt for cnt in contours if 1000 < cv.contourArea(cnt) < 100000], key=cv.contourArea, reverse=True)

    # Display contour so user can see what is being tracked 
    cv.drawContours(frame, contours, -1, (0,255,0), 2)
    
    # Applies If loop is contour is true and found 
    if contours:
        # Sets the contour found as UNO card 
        max_contour = contours[0]
        # Calculates the bounding box for contour found
        x, y, w, h = cv.boundingRect(max_contour)
        # Stores ROI based on bounding box frame 
        roi = frame[y:y+h, x:x+w]
        
        # Display bounding box 
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # ROI found and set to true 
        return roi, True
    
    # No ROI found and set to false 
    return None, False

# Sets the holding size to 5 
hue_buffer_size = 5
# Holds 5 hue elements and removes the old ones 
hue_buffer = deque(maxlen=hue_buffer_size)

# Function to get most common or dominant hue 
def get_most_common_hue(hue_buffer):
    try:
        # First tries to find most common hue - mode 
        return mode(hue_buffer)
    except:
        # Otherwise, gets the last hue 
        return hue_buffer[-1]
    
# Predict color using pre-trained model and ROI found along with checking prediction with hue 
def check_prediction_confidence(model, roi, threshold=0.5):
    # Resize image to be set to (128, 128) - size required for model to be applie
    roi_resized = cv.resize(roi, (128, 128))
    # Normalize values by dividing by 255 to scale it down to either 0 or 1 which is black or white
    roi_normalized = roi_resized / 255.0
    # Expand dimension to allow for CNN per-trained model to be applied
    roi_expanded = np.expand_dims(roi_normalized, axis=0)

    # Applies the model to get prediction result
    prediction = model.predict(roi_expanded)
    # Gets the predicted label 
    prediction_index = np.argmax(prediction, axis=1)[0]
    # Stores the predicted label in variable 
    confidence = np.max(prediction)

    # Convert image to HSV as it was converted to gray scale in process_frame function 
    roi_hsv = cv.cvtColor(roi_resized, cv.COLOR_BGR2HSV)
    # Splits HSV values individually 
    hue, saturation, value = cv.split(roi_hsv)
    # Calculates the histogram -  Pixel values frequency distribution set to range from 0 to 180 
    hist, _ = np.histogram(hue, bins=180, range=[0,180])
    # Stores the dominate hue value from histogram 
    dominant_hue = np.argmax(hist)

    # Adds the dominate hue to buffer list 
    hue_buffer.append(dominant_hue)
    # Gets the most common hue 
    common_hue = get_most_common_hue(hue_buffer)

    # Sets the range of hue - Adjusted value through trial and error process 
    green_hue_range = range(65, 100)
    blue_hue_range = range(100, 140)
    # Having 2 red hue is due to red values are wrapped around in HSV circular scale 
    # so to ensure that darker and light red color is taken into account 2 hue values are given 
    red_hue_range_low = range(0, 10)
    red_hue_range_high = range(160, 180)

    # Checks if common hue is in blue range and check if predicted index is for blue 
    if common_hue in blue_hue_range and prediction_index != 0:
        # If it meets condition, sets the index to 0 which represent blue 
        prediction_index = 0
        # Set confidence level to 1 as it matches requirement 
        confidence = 1.0  
    # Checks if common hue is in green range and check if predicted index is for green 
    elif common_hue in green_hue_range and prediction_index != 1:
        # If it meets condition, sets the index to 1 which represent green 
        prediction_index = 1
        # Set confidence level to 1 as it matches requirement 
        confidence = 1.0
    # Checks if common hue is in red range and check if predicted index is for red 
    elif (common_hue in red_hue_range_low or common_hue in red_hue_range_high) and prediction_index != 2:
        # If it meets condition, sets the index to 2 which represent red 
        prediction_index = 2
        # Set confidence level to 1 as it matches requirement 
        confidence = 1.0  

    # Stores the final prediction 
    color_name = index_to_color[prediction_index]
    # Prints the final value of terminal 
    print(f"Dominant Hue: {dominant_hue}, Most Common Hue: {common_hue}, Detected color: {color_name} with confidence: {confidence}")

    # If confidence is below the threshold 
    if confidence < threshold:
        # Prints confidences level and inform user that there issue resulting in ROI requiring to be checked 
        print(f"Low confidence: {confidence}, recheck ROI")
        # Returns no prediction index 
        return None
    # Otherwise, if confidence is above threshold, return the prediction index 
    return prediction_index

# Sets color name to index 
index_to_color = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow'}

# Sets path of where pre-trained model is stored 
model_path = 'C:/Users/bhati/Python Code/IS-CW2/uno_colorG_model.keras'  # Adjust the path as necessary
# Loads pre-trained model
model = load_model(model_path)

# Initialize capture of video stream 
cap = cv.VideoCapture(0)

# Main loop for color recognition 
while True:
    # Capture the frame from webcam and is held in frame variable 
	# ret means boolean to indicate whether frame is captured or not
    ret, frame = cap.read()
    # if ret is true and frame is captured 
    if ret:
        # Applies function to process frame and capture ROI 
        roi, roi_detected = process_frame(frame)
        # If ROI is captured and set to true 
        if roi_detected:
            # Prints confirmation message 
            print("ROI Detected")
            # Applies function to do color recognition using pre-trained model and hue 
            color_index = check_prediction_confidence(model, roi)
            # If prediction has been made from function 
            if color_index is not None:
                # Store the color according to index values set
                color_name = index_to_color[color_index]
                # Prints the color recognised 
                print(f"Detected color: {color_name}") 
                # Write color name on frame
                cv.putText(frame, color_name, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Prints confirmation message 
                print("Color name should now be on frame") 
            else:
                # If no prediction is made, print message to inform user 
                print("Color index is None") 
        else:
            # If no ROI is found, print message to inform user 
            print("ROI not detected")  

        # If ROI is found 
        if roi_detected:
            # Display ROI in window 
            cv.imshow('ROI', roi)
        # and display video stream frame 
        cv.imshow('Frame', frame)

    # if q button is pressed 
    if cv.waitKey(1) & 0xFF == ord('q'):
        # Break out of the loop
        break

# Release the webcam
cap.release()
# and close all the windows
cv.destroyAllWindows()
