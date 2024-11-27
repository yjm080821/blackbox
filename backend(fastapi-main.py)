from fastapi import FastAPI, File, UploadFile
import uvicorn
import cv2
import os
import numpy as np

app = FastAPI()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    anomaly_score = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        anomaly_score += np.random.rand() * 0.1

    cap.release()
    anomaly_score = min(anomaly_score / frame_count, 1.0)
    if anomaly_score > 0.7:
        return {"status": "이상", "score": anomaly_score}
    elif anomaly_score > 0.4:
        return {"status": "흡연", "score": anomaly_score}
    else:
        return {"status": "정상", "score": anomaly_score}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    result = analyze_video(file_path)
    return {"filename": file.filename, "result": result}


@app.post("/analyze")
async def analyze_existing_video(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "파일이 존재하지 않습니다."}

    result = analyze_video(file_path)
    return {"filename": filename, "result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
