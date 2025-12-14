import cv2

# IDを0から順にチェックしていく
for i in range(10): 
    # cv2.VideoCapture(i) でカメラを開こうとする
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(f"✅ カメラID {i}: 利用可能です。")
        # カメラを解放する
        cap.release()
    else:
        # IDが見つからなくなった時点で終了してもOK
        print(f"❌ カメラID {i}: 利用できません。")
        # break