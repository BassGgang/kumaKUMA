import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# FastAPIアプリケーションのインスタンスを作成
app = FastAPI()

# CORSミドルウェアを追加
# フロントエンドが http://localhost:8000 で動作することを想定
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLOv8モデルをロード（'n'はnanoモデルで最速・最軽量）
# 初回実行時にモデルが自動的にダウンロードされます
model = YOLO('best.pt')
print("YOLOv8 model loaded successfully.")

@app.get("/")
def read_root():
    return {"message": "YOLOv8 detection server is running."}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    画像ファイルを受け取り、YOLOv8で物体検出を実行して結果を返す
    """
    # アップロードされたファイルをメモリ内で読み込む
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # モデルで推論を実行
    results = model(image, conf=0.1)

    # 検出結果を整形
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 座標、信頼度、クラスIDを取得
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]

            # "person" と "car", "bear" のみ検出
            if class_name in ["person", "car", "bear"]:
                # フロントエンドが期待する形式に変換
                # bbox: [x, y, width, height]
                x1, y1, x2, y2 = xyxy
                bbox = [x1, y1, x2 - x1, y2 - y1]

                detections.append({
                    "class": class_name,
                    "score": conf,
                    "bbox": bbox
                })

    return {"predictions": detections}

# サーバーを起動するためのコマンド（ターミナルで直接実行する場合）
# uvicorn main:app --reload --port 8001
