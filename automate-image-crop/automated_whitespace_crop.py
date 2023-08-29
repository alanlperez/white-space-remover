import os
import cv2

folder_path = 'output_folder 11'
images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for image_name in images:
    # Read the image
    img_path = os.path.join(folder_path, image_name)
    img = cv2.imread(img_path)

    print(img_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to convert the image to binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find the contours in the image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Combine all bounding boxes
    x, y, w, h = bounding_boxes[0]
    for bbox in bounding_boxes[1:]:
        x = min(x, bbox[0])
        y = min(y, bbox[1])
        w = max(w, bbox[2] + bbox[0] - x)
        h = max(h, bbox[3] + bbox[1] - y)

    # Crop the image to the combined bounding box
    cropped_img = img[y:y+h, x:x+w]

    if not os.path.exists('output_folder'):
        os.makedirs('output_folder')

    # Save the cropped image
    cv2.imwrite(os.path.join('output_folder', image_name), cropped_img)

