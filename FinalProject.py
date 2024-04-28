import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load and display the grayscale lung CT image
image = cv2.imread(r"C:\Users\mizab\Desktop\UTRGV CSCI 4301\Final_project\Final_project\lung_ct.jpg", cv2.IMREAD_GRAYSCALE)

# Get the size of the image
M, N = image.shape
print("Image size (M x N):", M, "x", N)

# Compute 2D FFT
fft_image = np.fft.fft2(image)
# Shift the zero-frequency component to the center
fft_image_shifted = np.fft.fftshift(fft_image)

#Display images in subplots
##Lung ct original
plt.figure(figsize=(10, 5))
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray')
plt.title("lung CT")
#plt.xlabel('Image size (M x N):', [M], "x", [N])
#plt.axis('off')

##Magnitude
magnitude = np.log(1 + np.abs(fft_image_shifted))
plt.subplot(2, 3, 2)
plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude (Log Scale)'),plt.axis('off')

##Phase
phase = np.angle(fft_image_shifted)
plt.subplot(2, 3, 3)
plt.imshow(phase, cmap='gray')
plt.title('Phase Spectrum'), plt.axis('off')
#plt.show()

##Down-sampled image
downsampled_image = cv2.resize(image, (N//2, M//2))
plt.subplot(2, 3, 4)
plt.imshow(downsampled_image, cmap='gray')
plt.title('Down-sampled Image')
#plt.axis('off')

# Get frequency domain of down-sampled image
fft_downsampled = np.fft.fft2(downsampled_image)
## Zero-padding to interpolate in frequency domain
padded_fft = np.zeros_like(fft_image_shifted)
padded_fft[:fft_downsampled.shape[0], :fft_downsampled.shape[1]] = fft_downsampled
interpolated_image = np.fft.ifft2(padded_fft).real
plt.subplot(2, 3, 5)
plt.imshow(interpolated_image, cmap= 'gray')
plt.title("Interpolated Image")

#Spatial domain interpolated image
interpolated_image_spatial = cv2.resize(downsampled_image, (N, M), interpolation=cv2.INTER_LINEAR)
plt.subplot(2, 3, 6)
plt.imshow(interpolated_image_spatial, cmap='gray')
plt.title("spatial domain interpolation."),plt.axis('off')
plt.show()
