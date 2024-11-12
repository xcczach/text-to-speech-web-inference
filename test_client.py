import requests
import numpy as np
import soundfile as sf
import io

# 设置请求 URL 和请求体
url = "http://localhost:9234/test"
data = {"text": "Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via programming."}

# 发送 POST 请求，并启用流式传输
response = requests.post(url, json=data, stream=True)

# 检查响应状态码
if response.status_code == 200:
    audio_chunks = []  # 用于保存分块的音频数据

    buffer = io.BytesIO()  # 用于保存完整的音频数据    
    for chunk in response.iter_content(chunk_size=4096):
        buffer.write(chunk)

    buffer.seek(0)
    # 将所有块合并为一个完整的numpy数组
    audio_data = np.load(buffer)

    # 使用 soundfile 写入音频文件，指定采样率（例如，44100Hz）
    sample_rate = 24000  # 请根据实际情况调整
    sf.write("output_audio.wav", audio_data, sample_rate)

    print("音频数据已保存为 output_audio.wav")
else:
    print(f"请求失败，状态码：{response.status_code}")
