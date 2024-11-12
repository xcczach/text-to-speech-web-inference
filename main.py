from ml_web_inference import expose, Request, StreamingResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import io
import numpy as np

model = None
config = None

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

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
    model.to(get_device())

def hangup():
    global model
    del model
    torch.cuda.empty_cache()


expose(
    "test",
    inference,
    port=9234,
    hangup_timeout_sec=10,
    hangup_interval_sec=5,
    init_function=init,
    hangup_function=hangup,
)