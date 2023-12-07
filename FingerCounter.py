import cv2
import time
import HandTrackingModule as htm

# Tentukan resolusi kamera
wCam, hCam = 640, 480

# Ambil video dari webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Inisialisasi waktu frame sebelumnya
pTime = 0

# Buat objek detektor tangan
detector = htm.handDetector(detectionCon=0.75)

# Tentukan ID ujung jari
tipIds = [4, 8, 12, 16, 20]

# Mulai loop pemrosesan video
while True:
    # Baca frame dari webcam
    success, img = cap.read()

    # Deteksi tangan dalam frame
    img, hand_type = detector.findHands(img)

    # Dapatkan landmark (titik kunci) tangan
    lmList = detector.findPosition(img, draw=False)

    img = cv2.flip(img, 1)


    # Periksa apakah ada tangan yang terdeteksi
    if len(lmList) != 0:
        # Inisialisasi list jari kosong
        fingers = []

        # Periksa posisi ibu jari
        if (lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]) and hand_type == "Left":
            fingers.append(1)
        elif (lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]) and hand_type == "Right" :
            fingers.append(0)
        elif (lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]) and hand_type == "Right" :
            fingers.append(1)
        elif (lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]) and hand_type == "Left" :
            fingers.append(0)
       

        # Periksa posisi 4 jari lainnya
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        hand_type = "Right" if hand_type == "Left" else "Left"
        # Hitung total jumlah jari yang diangkat
        totalFingers = fingers.count(1)
        # Tampilkan total jari yang diangkat
        print(f"{hand_type}: {totalFingers} fingers")

        # Gambar persegi untuk area penghitungan
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{hand_type} Hand", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # Tampilkan total jari yang diangkat dalam persegi
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                   10, (255, 0, 0), 25)

    # Hitung FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Tampilkan FPS pada frame
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    # Tampilkan frame
    cv2.imshow("Image", img)

    # Periksa penekanan tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup semua jendela
cv2.destroyAllWindows()
