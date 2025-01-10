import cv2
import numpy as np

'''
DETECT diameter/perimeter (pixel size) given a circle with the fixed distance.
'''


import cv2

# Callback function for cropping


def crop_selected_img(image):
    '''
    Select image and crop via opencv roi toolbox
    '''
    image_copy = image.copy()
    ref_point = []
    cropping = False
    resize_idx = 4
    image_resize = cv2.resize(image,(image.shape[1]//resize_idx,image.shape[0]//resize_idx))


    while True:
        # Display the image and let the user select a region
        r = cv2.selectROI("Image", image_resize)
        x1, y1, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])

        # reproject to origional size:
        x2, y2 = x1 + w, y1 + h
        x1,y1,x2,y2 = int(x1*resize_idx),int(y1*resize_idx),int(x2*resize_idx),int(y2*resize_idx)

        if w > 0 and h > 0:
            
            roi = image[y1:y2, x1:x2]
            cv2.imshow("Processed ROI", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break


    
    return roi



def find_coin_diameter(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # denoise image
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # Gaussian Blur
    # gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # # Median Blur
    # median_blur = cv2.medianBlur(image, 5)


    image = crop_selected_img(image)

    # Brighten the image
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)  # Increase alpha and beta for brightness
    # Non-Local Means Denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ================PREVIOUS METHOD: Find edges using edge detection ===============
    edges = cv2.Canny(blurred, 20, 250)
    # Find the contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ================OTHER METHOD: FIND CIRCLE WITH BLUE COLOR======================
    # # Convert the image to HSV for better color segmentation
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # Define the blue color range in HSV
    # lower_blue = np.array([100, 150, 50])  # Adjust these values if necessary
    # upper_blue = np.array([140, 255, 255])
    # # Create a binary mask where blue is white and other colors are black
    # mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    # # Find contours in the mask
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    
    
    contours_cnt = [len(x) for x in contours]
    perimeter = cv2.arcLength(contours[np.argmax(contours_cnt)], True)
    # Draw contours
    output_image = image.copy()
    cv2.drawContours(output_image, [contours[np.argmax(contours_cnt)]], -1, (0, 255, 0), 2)  # Draw in green with thickness 2

    # Find the largest contour (which should be the coin)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the minimum enclosing circle of the largest contour
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    # Calculate the diameter of the coin
    diameter = 2 * radius

    return diameter, perimeter, output_image

# Example usage
image_paths = [  
    # "/media/anranli/DATA/data/fish/Growth Study Day 3 [12-18-24]/Calibration Dot.JPG",
    # "/media/anranli/DATA/data/fish/Growth Study Day 2 [12-11-24]/Calibration Dot.JPG",
    "/media/anranli/DATA/data/fish/Growth Study Day 4 [12-30-24]/Calibration Dot.JPG",
    "/media/anranli/DATA/data/fish/Tk 4 - varied data/Calibration Dot.JPG",
    # "/media/anranli/DATA/data/fish/Tk 5 - varied data/Calibration Dot.JPG"
    ]
for image_path in image_paths:
    diameter,perimeter, out_img = find_coin_diameter(image_path)
    print(f'\nImage Name: {image_path}')
    print("Diameter of the circle:{}, Perimeter: {} ".format(diameter,perimeter))
    print(f'pi={perimeter/diameter}')
    print(f'Unit pixel size = {1.27/diameter} cm/pixel')
    print(20*'==')
    cv2.imshow("Fish Boundary", out_img)

    # Wait and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()