import requests
import soundfile as sf
import io

url = "http://localhost:9234/tts"
# data = {"text": "Chtholly Nota Seniorious is the main female protagonist of the light novel series \"WorldEnd: What do you do at the end of the world? Are you busy? Will you save us?\" She is a Leprechaun fairy soldier and the wielder of the powerful holy sword \"Seniorious.\" Throughout the story, Chtholly develops a deep bond with the male lead, Willem Kmetsch, and faces numerous challenges, including the erosion of her past memories."}
data = {"text": "珂朵莉是世界上最幸福的女孩"}

response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    audio_chunks = []

    buffer = io.BytesIO()
    for chunk in response.iter_content(chunk_size=4096):
        buffer.write(chunk)

    buffer.seek(0)
    audio_data, sample_rate = sf.read(buffer)
    sf.write("output_audio.wav", audio_data, sample_rate)

    print("Audio saved at output_audio.wav")
else:
    print(f"Bad request：{response.status_code}")
