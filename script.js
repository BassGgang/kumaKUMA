// --- DOM要素の取得 ---
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const loadingDiv = document.getElementById('loading');
const warningDiv = document.getElementById('warning');
const alertSound = document.getElementById('alert-sound');
const debugView = document.getElementById('debug-view');
const modelSelector = document.getElementById('model-selector');
const startStopBtn = document.getElementById('start-stop-btn');
const yoloStatusDiv = document.getElementById('yolo-status');

// --- 状態管理変数 ---
let cocoModel = null;
let isDetecting = false;
let videoStream = null;
const YOLO_API_URL = 'http://localhost:8001/detect';
const YOLO_CHECK_URL = 'http://localhost:8001/';

// --- 初期化処理 ---
document.addEventListener('DOMContentLoaded', checkYoloServer);

// --- イベントリスナー ---
startStopBtn.addEventListener('click', toggleDetection);

// --- 関数定義 ---

/**
 * YOLOv8サーバーがオンラインかチェックし、UIを更新する
 */
async function checkYoloServer() {
    try {
        const response = await fetch(YOLO_CHECK_URL);
        if (response.ok) {
            yoloStatusDiv.textContent = 'YOLOv8サーバー: オンライン';
            yoloStatusDiv.className = 'control-item yolo-status-online';
        } else {
            throw new Error('Server not ready');
        }
    } catch (error) {
        yoloStatusDiv.textContent = 'YOLOv8サーバー: オフライン';
        yoloStatusDiv.className = 'control-item yolo-status-offline';
        console.warn('YOLOv8 server is not running. High-accuracy mode will not be available.');
    }
}

/**
 * 検出の開始・停止を切り替える
 */
async function toggleDetection() {
    if (isDetecting) {
        // 検出を停止
        isDetecting = false;
        startStopBtn.textContent = '検出を開始';
        startStopBtn.classList.remove('running');
        
        // カメラを停止
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        // 描画をクリア
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        debugView.innerText = '';

    } else {
        // 検出を開始
        startStopBtn.textContent = '検出を停止';
        startStopBtn.classList.add('running');
        isDetecting = true;
        
        try {
            // カメラをセットアップ
            await setupCamera();
            
            const modelType = modelSelector.value;
            if (modelType === 'coco-ssd' && !cocoModel) {
                loadingDiv.innerText = 'COCO-SSDモデルを読み込んでいます...';
                loadingDiv.style.display = 'block';
                cocoModel = await cocoSsd.load();
                loadingDiv.style.display = 'none';
            }
            
            // 検出ループを開始
            detectFrame();

        } catch (error) {
            console.error("カメラの起動またはモデルの読み込みに失敗しました:", error);
            isDetecting = false;
            startStopBtn.textContent = '検出を開始';
            startStopBtn.classList.remove('running');
        }
    }
}

/**
 * カメラをセットアップする
 */
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('お使いのブラウザではカメラ機能がサポートされていません。');
    }
    videoStream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
    });
    video.srcObject = videoStream;
    return new Promise(resolve => {
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve();
        };
    });
}

/**
 * 検出のメインループ
 */
async function detectFrame() {
    if (!isDetecting) return;

    const modelType = modelSelector.value;
    let predictions = [];

    if (modelType === 'coco-ssd') {
        if (!cocoModel) { // モデルがまだロードされていない場合
            console.log("COCO-SSDモデルの読み込みを待っています...");
            setTimeout(detectFrame, 500); // 少し待ってリトライ
            return;
        }
        predictions = await cocoModel.detect(video);
        renderPredictions(predictions);
        requestAnimationFrame(detectFrame); // 次のフレームへ

    } else if (modelType === 'yolov8') {
        try {
            predictions = await detectWithYolo();
        } catch (error) {
            console.error("YOLOv8での検出エラー:", error);
            debugView.innerText = 'YOLOv8サーバーとの通信に失敗しました。';
            // サーバーが停止した場合に備え、再度チェック
            await checkYoloServer();
        }
        renderPredictions(predictions);
        setTimeout(detectFrame, 250); // 1秒に4回程度の間隔で実行
    }
}

/**
 * YOLOv8サーバーと通信して検出を行う
 */
async function detectWithYolo() {
    // ビデオフレームをキャンバスに描画
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    return new Promise((resolve, reject) => {
        tempCanvas.toBlob(async (blob) => {
            if (!blob) return reject(new Error('Canvas to Blob conversion failed.'));
            
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            try {
                const response = await fetch(YOLO_API_URL, {
                    method: 'POST',
                    body: formData,
                });
                if (!response.ok) {
                    throw new Error(`Server responded with status ${response.status}`);
                }
                const data = await response.json();
                resolve(data.predictions || []);
            } catch (error) {
                reject(error);
            }
        }, 'image/jpeg');
    });
}


/**
 * 検出結果を画面に描画する
 */
function renderPredictions(predictions) {
    // デバッグビューの更新
    if (predictions.length > 0) {
        debugView.innerText = predictions.map(p => `${p.class} (${Math.round(p.score * 100)}%)`).join('\n');
    } else {
        debugView.innerText = '何も検出されていません...';
    }

    // キャンバスの描画
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let bearDetected = false;

    predictions.forEach(prediction => {
        if (prediction.class === 'bear') {
            bearDetected = true;
            const [x, y, width, height] = prediction.bbox;
            
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, width, height);
            
            ctx.fillStyle = 'red';
            ctx.font = '24px sans-serif';
            const text = `クマ (${Math.round(prediction.score * 100)}%)`;
            ctx.fillText(text, x, y > 20 ? y - 5 : 20);
        }
    });

    // 警告の表示・再生
    if (bearDetected) {
        warningDiv.style.display = 'block';
        alertSound.play().catch(e => console.warn("音声の自動再生がブロックされました。"));
    } else {
        warningDiv.style.display = 'none';
    }
}
