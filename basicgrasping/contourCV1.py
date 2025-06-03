import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2 # Computer vision library
# Read the color image
image = cv2.imread("Image1.jpeg")

# Make a copy
new_image = image.copy()
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Display the grayscale image
# cv2.imshow('Gray image', gray)  
# Wait for keypress to continue
#cv2.destroyAllWindows() # Close windows
 
# Convert the grayscale image to binary
ret, binary = cv2.threshold(gray, 100, 255, 
  cv2.THRESH_OTSU)
# Display the binary image
# cv2.imshow('Binary image', binary)
# Wait for keypress to continue
# cv2.destroyAllWindows() # Close windows
# To detect object contours, we want a black background and a white 
# foreground, so we invert the image (i.e. 255 - pixel value)
inverted_binary = ~binary
# cv2.imshow('Inverted binary image', inverted_binary)
# Wait for keypress to continue
# cv2.destroyAllWindows() # Close windows
# Find the contours on the inverted binary image, and store them in a list
# Contours are drawn around white blobs.
# hierarchy variable contains info on the relationship between the contours
contours, hierarchy = cv2.findContours(inverted_binary,
  cv2.RETR_TREE,
  cv2.CHAIN_APPROX_SIMPLE)
# Draw a bounding box around all contours
for c in contours:
  x, y, w, h = cv2.boundingRect(c)
  
# Make sure contour area is large enough
  if (cv2.contourArea(c)) > 2000 and ((cv2.contourArea(c)) < 500000):
    # cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255,255,255), 5)
    print("Coordinates:", x, y, w, h)
    
    fig, ax = plt.subplots()
    rect = matplotlib.patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.imshow(image)
    plt.show()
# cv2.imshow('All contours with bounding box', with_contours)
# cv2.waitKey(0)