# tts-web-inference

Provide a web interface for text-to-speech inference.

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the TTS checkpoint:
```bash
export HF_ENDPOINT="https://hf-mirror.com" # Optional, necessary if you are in China
python ckpts/download.py
```

Start the server and wait for a few seconds for the server to be ready:
```bash
bash run.sh
```
You can customize some parameters in `run.sh`.

Run `test_client.py` for an example post!
```bash
python test_client.py
```