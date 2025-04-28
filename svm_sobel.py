import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar dengan OpenCV
image = cv2.imread('D:/UNIOR/DSC_0146.JPG')  # Ganti dengan path gambar Anda

# Memastikan gambar berhasil dibaca
if image is None:
    print("Gambar tidak ditemukan!")
    exit()

# Mengonversi gambar ke grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Jika gambar tidak ditemukan
if image is None:
    print("Gambar tidak ditemukan!")
    exit()

# Menggunakan Sobel untuk mendeteksi tepi horizontal (Gx) dan vertikal (Gy)
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradien pada arah X (horizontal)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradien pada arah Y (vertikal)

# Menghitung magnitudo gradien
magnitude = cv2.magnitude(sobel_x, sobel_y)

# Menghitung arah gradien (optional)
angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)

# Menampilkan hasil deteksi tepi
plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.imshow(sobel_x, cmap='gray')
plt.title('Tepi Horizontal (Gx)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_y, cmap='gray')
plt.title('Tepi Vertikal (Gy)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(magnitude, cmap='gray')
plt.title('Magnitudo Gradien')
plt.axis('off')

plt.show()
