import cv2
import matplotlib.pyplot as plt

# Membaca gambar dengan OpenCV
image = cv2.imread('D:/UNIOR/DSC_0146.JPG')  # Ganti dengan path gambar Anda

# Memastikan gambar berhasil dibaca
if image is None:
    print("Gambar tidak ditemukan!")
    exit()

# Mengonversi gambar ke grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Menampilkan gambar asli dan gambar grayscale
plt.figure(figsize=(10, 5))

# Gambar asli
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Mengubah BGR ke RGB untuk ditampilkan dengan benar
plt.title('Gambar Asli')
plt.axis('off')

# Gambar grayscale
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')  # Menampilkan gambar grayscale
plt.title('Gambar Grayscale')
plt.axis('off')

plt.show()
