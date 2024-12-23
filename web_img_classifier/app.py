from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import requests
import logging

app = FastAPI()

# 设置允许的跨域来源
origins = [
    "http://localhost",  # 允许来自 localhost 的请求
    "http://127.0.0.1",  # 允许来自 127.0.0.1 的请求
    "http://localhost:8000",  # 允许来自本地 FastAPI 前端的请求
    "http://127.0.0.1:8000",
]

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的跨域来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)


# 配置日志
logging.basicConfig(level=logging.INFO)

# 加载预训练的MobileNetV2模型
model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")

# 加载标签
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
response = requests.get(LABELS_URL)
class_names = response.text.splitlines()

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image_bytes):
    image = preprocess_image(image_bytes)
    predictions = model(image)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]
    return predicted_label

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    logging.info("Received request")
    image_bytes = await file.read()
    try:
        predicted_label = predict(image_bytes)
        logging.info(f"Prediction: {predicted_label}")
        print(type(predicted_label))
        print(predicted_label)
        return JSONResponse(content={"label": predicted_label})
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
