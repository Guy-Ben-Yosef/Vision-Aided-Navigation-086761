import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------- (a)  -----------#

# Read the images
image1 = cv2.imread('img1.jpeg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('img2.jpeg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect SIFT features and compute descriptors.
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Draw keypoints on the images
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None)

# Plot the images with keypoints
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.title('SIFT Keypoints Image 1')
plt.imshow(image1_with_keypoints)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('SIFT Keypoints Image 2')
plt.imshow(image2_with_keypoints)
plt.axis('off')

plt.show()

# ----------- (b)  -----------#

# Detect keypoints and descriptors for b only
keypoints, descriptors = sift.detectAndCompute(image1, None)

# Choose a representative keypoint
representative_keypoint = keypoints[0]  #  taking the first keypoint

# Draw the representative keypoint
image_with_keypoint = cv2.drawKeypoints(image1, [representative_keypoint], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show the image with the representative keypoint
plt.imshow(image_with_keypoint)
plt.title(f'Keypoint: Scale={representative_keypoint.size}, Orientation={np.degrees(representative_keypoint.angle):.2f} degrees')
plt.show()

# Print out the scale and orientation of the representative keypoint
print(f'Scale (size): {representative_keypoint.size}')
print(f'Orientation (angle in degrees): {np.degrees(representative_keypoint.angle)}')

# ----------- (c)  -----------#

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Match descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to find good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Sort matches based on the distance
good_matches = sorted(good_matches, key=lambda x: x.distance)

# Draw the best matches (inliers)
img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches[:10], None, flags=2)

# Use RANSAC to find inliers and outliers
if len(good_matches) >= 4:
    # Prepare the data for findHomography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Compute the homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Convert mask to a list of integers for OpenCV compatibility
    matches_mask = mask.ravel().tolist()
    inlier_matches = [good_matches[i] for i in range(len(matches_mask)) if matches_mask[i]]

    # Draw inliers
    img_inliers = cv2.drawMatches(image1, keypoints1, image2, keypoints2, inlier_matches, None, matchColor=(0, 255, 0), singlePointColor=None, matchesMask=None, flags=2)

    # Draw outliers
    outlier_matches = [good_matches[i] for i in range(len(matches_mask)) if not matches_mask[i]]
    img_outliers = cv2.drawMatches(image1, keypoints1, image2, keypoints2, outlier_matches, None, matchColor=(0, 0, 255), singlePointColor=None, matchesMask=None, flags=2)

    # Display the results
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(img_inliers)
    plt.title('Inliers (Good Matches)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_outliers)
    plt.title('Outliers (Bad Matches)')
    plt.axis('off')

    plt.show()
