import requests

# 图像路径
image_path = 'D:\\car.jpg'

# 发送请求
url = 'http://127.0.0.1:8000/predict'
with open(image_path, 'rb') as image_file:
    response = requests.post(url, files={'file': image_file})

# 打印响应
print(f"Response status code: {response.status_code}")
print(response.json())
