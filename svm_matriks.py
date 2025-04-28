import cv2
import numpy as np

# Membaca gambar menggunakan OpenCV
image = cv2.imread('D:/UNIOR/DSC_0147.JPG', cv2.IMREAD_COLOR)  # Ganti dengan path gambar Anda

# Menampilkan gambar
# cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Menampilkan data gambar dalam bentuk matriks (array)
print("Data gambar dalam bentuk matriks:")
print(image)

# Jika ingin mengetahui dimensi gambar
print("Dimensi gambar:", image.shape)
