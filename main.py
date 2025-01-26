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


import re


def _detect_language(text):
    counts = {
        "en": len(re.findall(r"[a-zA-Z]", text)),
        "zh": len(re.findall(r"[\u4e00-\u9fff]", text)),
        "ja": len(re.findall(r"[\u3040-\u30ff]", text)),
    }
    max_lang = max(counts, key=counts.get)
    return max_lang if counts[max_lang] > 0 else "en"


def _unit_len(text: str):
    # count the number of characters of Chinese and Japanese, and the number of words of English; add them together
    return (
        len(re.findall(r"[\u4e00-\u9fff]", text))
        + len(re.findall(r"[\u3040-\u30ff]", text))
        + len(text.split())
    )


def _merge_short_sentences(sentences: list, max_units: int):
    merged_segments = []
    temp_segment = ""
    for i in range(0, len(sentences)):
        if _unit_len(temp_segment) + _unit_len(sentences[i]) <= max_units:
            temp_segment += sentences[i]
        else:
            if temp_segment:
                merged_segments.append(temp_segment)
            temp_segment = sentences[i]
    if temp_segment:
        merged_segments.append(temp_segment)
    return merged_segments


def _split_by_punctuation(text: str, puncs: str):
    sentences = re.split(rf"(?<=[{puncs}])", text)
    return [s.strip() for s in sentences if s.strip()]


def _split_text_for_tts(text: str, max_units: int):
    # split text into segments with 。！？….!?
    sentences = _split_by_punctuation(text, "。！？….!?")
    # merge short sentences until the segment is less than max_units
    merged_segments = _merge_short_sentences(sentences, max_units)
    # split the segment with ,，:：;；、 if sentence is longer than max_units
    splited_segments = []
    for segment in merged_segments:
        if _unit_len(segment) > max_units:
            new_segments = _split_by_punctuation(segment, ",，:：;；、")
            splited_segments.extend(_merge_short_sentences(new_segments, max_units))
        else:
            splited_segments.append(segment)
    return splited_segments


async def inference(request: Request) -> StreamingResponse:
    data = await request.json()
    text = data["text"]
    text_segments = _split_text_for_tts(text, 20)
    result_arrs = []
    for segment in text_segments:
        lang = _detect_language(segment)
        result_dict = model.synthesize(
            segment,
            config,
            speaker_wav=lang_to_sample_path[lang],
            language=lang,
            gpt_cond_len=3,
        )
        result_arr = torch.tensor(result_dict["wav"])
        result_arrs.append(result_arr)
    result_arr = torch.cat(result_arrs)
    result = io.BytesIO()
    torchaudio.save(result, result_arr.unsqueeze(0), 24000, format="wav")
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
