from ml_web_inference import expose, Request, StreamingResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import io
import re
import argparse
import torchaudio
import setproctitle

model = None
config = None

lang_to_sample_path = {
    "en": "en_sample.wav",
    "zh": "zh-cn-sample.wav",
    "ja": "ja-sample.wav",
}


def detect_language(text):
    if re.search(r"[a-zA-Z]", text):
        return "en"
    elif re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    elif re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    else:
        return "unknown"


async def inference(request: Request) -> StreamingResponse:
    data = await request.json()
    text = data["text"]
    lang = detect_language(text)
    result_dict = model.synthesize(
        text,
        config,
        speaker_wav=lang_to_sample_path[lang],
        language=lang,
        gpt_cond_len=3,
    )
    print(result_dict.keys())
    result_arr = result_dict["wav"]
    result = io.BytesIO()
    torchaudio.save(result, torch.tensor(result_arr).unsqueeze(0), 24000, format="wav")
    result.seek(0)
    return StreamingResponse(result, media_type="application/octet-stream")


def init():
    global model, config
    if config is None:
        config = XttsConfig()
        config.load_json("ckpts/xttsv2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="ckpts/xttsv2", eval=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")


def hangup():
    global model
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    setproctitle.setproctitle("tts-web-inference")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9234)
    parser.add_argument("--api-name", type=str, default="tts")
    parser.add_argument("--hangup-timeout-sec", type=int, default=900)
    parser.add_argument("--hangup-interval-sec", type=int, default=60)
    args = parser.parse_args()
    expose(
        args.api_name,
        inference,
        port=args.port,
        hangup_timeout_sec=args.hangup_timeout_sec,
        hangup_interval_sec=args.hangup_interval_sec,
        init_function=init,
        hangup_function=hangup,
    )
