import cv2
import glob
import time
r = 1
for avi_name in glob.glob(f'C:/Users/oosim/Desktop/avi/{r}_*.avi'):
    print(avi_name)
cap = cv2.VideoCapture(avi_name)

#動画の表示
while (cap.isOpened()):
    #フレーム画像の取得
    ret, frame = cap.read()
    #画像の表示
    cv2.imshow("Image", frame)
    #キー入力で終了
    if cv2.waitKey(10) != -1:
        break

cap.release()
cv2.destroyAllWindows()