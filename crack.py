import cv2
import numpy as np

def detect_blemishes(image_path):
    image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = image.copy()
    cv2.drawContours(result_image, [contour for contour in contours], -1, (0, 255, 0), 2)
    
    total_pixels = image.shape[0]*image.shape[1]
    corrosion_area = sum(cv2.contourArea(contour) for contour in contours)
    corrosion_ratio = (corrosion_area*2 / total_pixels )

    if corrosion_ratio >= 0.9:
        message = "Heavily cracked"
    elif corrosion_ratio >= 0.6:
        message = "Moderately cracked"
    elif corrosion_ratio > 0:
        message = "Slightly cracked"
    else:
        message = "No significant cracks"
    
    cv2.putText(result_image, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 20, 10), 3)
    
    cv2.imshow('Corrosion Detection Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = r"pipeline-corrosion.jpg"
detect_blemishes(image_path)
