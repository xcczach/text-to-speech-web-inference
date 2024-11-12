from ml_web_inference import expose, Request, StreamingResponse, get_proper_device
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import io
import numpy as np
import argparse

model = None
config = None

async def inference(request: Request) -> StreamingResponse:
    data = await request.json()
    text = data["text"]
    result_arr = model.synthesize(
        text,
        config,
        speaker_wav="en_sample.wav",
        language="en",
        gpt_cond_len=3
    )["wav"]
    result = io.BytesIO()
    np.save(result, result_arr)
    result.seek(0)
    return StreamingResponse(result, media_type="application/octet-stream")


def init():
    global model, config
    if config is None:
        config = XttsConfig()
        config.load_json("ckpts/xttsv2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="ckpts/xttsv2", eval=True)
    model.to(get_proper_device(2000))

def hangup():
    global model
    del model
    torch.cuda.empty_cache()


if  __name__ == "__main__":
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