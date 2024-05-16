import cv2
import numpy as np

# Function for image pre-processing
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    median = cv2.medianBlur(equalized, 5)
    
    return median

# Function for image segmentation with color weighting
def segment_image(image):
    brown_lower = np.array([0,10, 10], dtype=np.uint8)
    brown_upper = np.array([20, 255, 255], dtype=np.uint8)
    orange_lower = np.array([15, 100, 100], dtype=np.uint8)
    orange_upper = np.array([25, 255, 255], dtype=np.uint8)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Threshold the image to extract regions with specified colors
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    
    damaged_mask = cv2.bitwise_or(brown_mask, orange_mask)
    
    kernel = np.ones((5,5), np.uint8)
    damaged_mask = cv2.morphologyEx(damaged_mask, cv2.MORPH_CLOSE, kernel)
    
    return damaged_mask



def calculate_damage_percentage(segmented_image, temperature, humidity):
    # Calculate the proportion of damaged area
    white_pixels = np.sum(segmented_image == 255)
    total_pixels = segmented_image.shape[0] * segmented_image.shape[1]
    damage_percentage = (white_pixels / total_pixels) * 100
    
    temperature_weight = 0.2 * (temperature - 32) / 5  # Adjust for Fahrenheit to Celsius
    humidity_weight = 0.2 * (humidity - 55) / 10
    
    severe_damage_weight = 0.55 * min(1, damage_percentage/100)  # Example: Linear function
    
    # Combine the weighted factors to calculate the damage percentage
    final_damage_percentage = temperature_weight + humidity_weight + severe_damage_weight
    
    return min(99.999,final_damage_percentage*100) # Convert to percentage



# Function to mark heavily damaged areas with contours
def mark_damaged_areas(image, original_image):
    # Find contours in the damaged area mask
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    cv2.drawContours(original_image, contours, -1, (0, 255, 255), 4)
    
    return original_image


image_path = r"pipeline-corrosion.jpg"
input_image = cv2.imread(image_path)

preprocessed_image = preprocess_image(input_image)

segmented_image = segment_image(input_image)

humidity = 70
temperature = 35

# Calculate percentage of damaged area
percentage_damaged_area = calculate_damage_percentage(segmented_image,temperature,humidity)

image_with_contours = mark_damaged_areas(segmented_image, input_image)


cv2.putText(image_with_contours, f"Damage Percentage: {percentage_damaged_area:.3f}%", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(60, 20, 10), 3)
cv2.imshow('Corrosion Detection', image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()