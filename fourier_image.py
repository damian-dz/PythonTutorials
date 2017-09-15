from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# open an 8bpp indexed image
img = Image.open('circles.bmp')
# load the image data into a numpy array
img_data = np.asarray(img)
# perform the 2-D fast Fourier transform on the image data
fourier = np.fft.fft2(img_data)
# move the zero-frequency component to the center of the Fourier spectrum
fourier = np.fft.fftshift(fourier)
# compute the magnitudes (absolute values) of the complex numbers
fourier = np.abs(fourier)
# compute the common logarithm of each value to reduce the dynamic range
fourier = np.log10(fourier)
# find the minimum value that is a finite number
lowest = np.nanmin(fourier[np.isfinite(fourier)])
# find the maximum value that is a finite number
highest = np.nanmax(fourier[np.isfinite(fourier)])
# calculate the original contrast range
original_range = highest - lowest
# normalize the Fourier image data ("stretch" the contrast)
norm_fourier = (fourier - lowest) / original_range * 255
# convert the normalized data into an image
norm_fourier_img = Image.fromarray(norm_fourier)

# display the original image and the Fourier image
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(img, cmap='gray')
ax2.imshow(norm_fourier_img)
ax1.title.set_text('Original Image')
ax2.title.set_text('Fourier Image')
plt.show()

# show the normalized Fourier image
norm_fourier_img.show()

# convert the output image to 8-bit pixels (grayscale) and save it
norm_fourier_img.convert('L').save('test.bmp')
