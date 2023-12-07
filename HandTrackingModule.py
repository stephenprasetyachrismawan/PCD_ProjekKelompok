"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""

import cv2
import mediapipe as mp
import time
import pyautogui 


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        # Inisialisasi kelas handDetector dengan parameter opsional
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Inisialisasi objek Mediapipe hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionCon, self.trackCon)
        # Inisialisasi objek Mediapipe drawing_utils
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyles = mp.solutions.drawing_styles

    def findHands(self, img, draw=True):
        # Mengubah format citra ke RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Proses deteksi tangan dengan Mediapipe hands
        self.results = self.hands.process(imgRGB)
        handedness = self.results.multi_handedness
        if handedness != None : 
            for idx, hand_handedness in enumerate(handedness):

                # print(hand_handedness.classification)
                # print(hand_handedness.classification[0].label)
                handedness = hand_handedness.classification[0].label

        # Menggambar landmark dan garis penghubung tangan jika draw=True
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS,
                                               self.mpDrawStyles.get_default_hand_landmarks_style(),
                                               self.mpDrawStyles.get_default_hand_connections_style() 
                                               )
        return img, handedness

    def findPosition(self, img, handNo=0, draw=True):
        # Mendapatkan posisi landmark tangan
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # handedness = self.results.multi_handedness
            for id, lm in enumerate(myHand.landmark):
                # Mendapatkan koordinat piksel (cx, cy) dari landmark
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Menambahkan informasi posisi landmark ke dalam lmList
                lmList.append([id, cx, cy])
                # Menggambar lingkaran di atas setiap landmark jika draw=True
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    # Menambahkan teks ID landmark di sampingnya
                    cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN,
                                1, (255, 255, 255), 2)
        return lmList

    
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        # Membaca frame dari kamera
        success, img = cap.read()
        # Mendeteksi dan melacak tangan pada frame
        img = detector.findHands(img)
        # Mendapatkan posisi landmark tangan
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            # Menampilkan posisi landmark jari ke-4 pada konsol
            print(lmList[4])

        # Mengukur FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Menampilkan FPS pada layar
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # Menampilkan frame pada layar
        cv2.imshow("Image", img)
        # Menunggu 1 milidetik, berhenti jika tombol 'q' ditekan
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
