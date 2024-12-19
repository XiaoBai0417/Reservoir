import cv2
import numpy as np
import random

# Generate a random binary image
def generate_random_binary_image(height, width, num_regions):
    binary_image = np.zeros((height, width), dtype=np.uint8)

    # Add rectangles
    for _ in range(num_regions // 10):
        x1, y1 = random.randint(0, width - 1), random.randint(0, height - 1)
        x2, y2 = random.randint(x1, width - 1), random.randint(y1, height - 1)
        binary_image[y1:y2, x1:x2] = 255

    # Add circles
    for _ in range(num_regions // 10):
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)
        radius = random.randint(5, 30)  # Small area circle
        cv2.circle(binary_image, (center_x, center_y), radius, 255, -1)

    # Add triangles
    for _ in range(num_regions // 10):
        pt1 = (random.randint(0, width - 1), random.randint(0, height - 1))
        pt2 = (random.randint(0, width - 1), random.randint(0, height - 1))
        pt3 = (random.randint(0, width - 1), random.randint(0, height - 1))
        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.drawContours(binary_image, [triangle_cnt], 0, 255, -1)

    return binary_image

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _, labeled_image = param
        component_id = labeled_image[y, x]
        if component_id > 0:  # Ignore background
            highlighted = np.zeros_like(labeled_image, dtype=np.uint8)
            highlighted[labeled_image == component_id] = 255
            cv2.imshow("Highlighted Component", highlighted)

if __name__ == "__main__":
    # Configuration
    height, width, num_regions = 400, 400, 10

    # Generate the image and connected components
    binary_image = generate_random_binary_image(height, width, num_regions)
    _, labeled_image = cv2.connectedComponents(binary_image, connectivity=8)

    print(f"Number of connected components: {np.max(labeled_image)} (excluding background)")

    # Display the image and set mouse callback
    cv2.imshow("Binary Image", binary_image)
    cv2.setMouseCallback("Binary Image", mouse_callback, (binary_image, labeled_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
