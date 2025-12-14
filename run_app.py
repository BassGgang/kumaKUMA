import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import threading
import time

# --- 設定 ---
MODEL_PATH = 'yolov8n.pt'  # 使用するモデルのパス

VIDEO_SOURCE = 0           # カメラのID (0は通常内蔵カメラ) masked 2025/12/11
# 変更点: カメラソースをRTSP URLに変更
# VIDEO_SOURCE = "rtsp://kaikuma:seikonojonta@111.89.122.117:554/stream1"

# VIDEO_SOURCE = 'media\sample.mp4'  # <--- ダウンロードしたビデオファイル名
ALERT_SOUND_PATH = 'alert.mp3' # 警告音のファイルパス
TARGET_CLASS = 'bear'      # 検出対象のクラス名
CONF_THRESHOLD = 0.5       # 検出の信頼度のしきい値
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = 'Bear Detection System'

# --- グローバル変数 ---
is_sound_playing = False
last_detection_time = 0
SOUND_COOLDOWN = 5 # 警告音のクールダウンタイム（秒）

# --- 関数定義 ---

def play_alert_sound():
    """警告音を別スレッドで再生する"""
    global is_sound_playing
    if not is_sound_playing:
        is_sound_playing = True
        try:
            playsound(ALERT_SOUND_PATH)
        except Exception as e:
            print(f"警告音の再生に失敗しました: {e}")
        is_sound_playing = False

def main():
    """メインの処理ループ"""
    global last_detection_time

    # モデルをロード
    print("YOLOv8モデルを読み込んでいます...")
    try:
        model = YOLO(MODEL_PATH)
        print("モデルの読み込みが完了しました。")
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return

    # カメラをキャプチャ
    print("カメラを起動しています...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"エラー: カメラ(ID:{VIDEO_SOURCE})を開けませんでした。")
        print("利用可能なカメラIDを確認するか、ビデオファイルのパスを指定してください。")
        return
    
    print("検出を開始します。ウィンドウを選択して 'q' キーを押すと終了します。")

    # ウィンドウを作成
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        # フレームを読み込む
        ret, frame = cap.read()
        if not ret:
            print("エラー: カメラからフレームを取得できませんでした。")
            break

        # モデルで推論を実行
        results = model(frame, verbose=False) # verbose=Falseでコンソール出力を抑制

        bear_detected = False
        # 検出結果を処理
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0].item()
                if conf < CONF_THRESHOLD:
                    continue

                cls_id = int(box.cls[0].item())
                class_name = model.names[cls_id]

                if class_name == TARGET_CLASS:
                    bear_detected = True
                    # バウンディングボックスの座標を取得
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # 矩形を描画
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # ラベルを描画
                    label = f"{class_name.upper()}: {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, FONT, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), FONT, 0.6, (255, 255, 255), 2)

        # クマが検出された場合の処理
        if bear_detected:
            # 警告テキストを表示
            cv2.putText(frame, "WARNING: BEAR DETECTED!", (50, 50), FONT, 1.2, (0, 0, 255), 3)
            
            # クールダウンを確認して警告音を再生
            current_time = time.time()
            if current_time - last_detection_time > SOUND_COOLDOWN:
                last_detection_time = current_time
                if not is_sound_playing:
                    threading.Thread(target=play_alert_sound, daemon=True).start()

        # 結果を表示
        cv2.imshow(WINDOW_NAME, frame)

        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 後処理
    print("アプリケーションを終了します。")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
