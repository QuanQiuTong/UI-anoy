# remote_server.py
import base64
import io
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "inference_log.jsonl"


# ======================
# 配置区
# ======================
MODEL_PATH = "/home/xiyuan/data/model/Qwen3-VL-4B-Instruct"
MAX_NEW_TOKENS = 1600

# ======================
# 初始化模型（只执行一次）
# ======================
print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model.eval()
print("Model loaded.")

# ======================
# FastAPI
# ======================
app = FastAPI(title="Remote VLM Inference Server")


class InferRequest(BaseModel):
    prompt: str
    image_base64: str


class InferResponse(BaseModel):
    text: str


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    print("Received inference request.")
    try:
        # Decode image
        image_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": req.prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        # Trim prompt tokens
        gen_ids = output_ids[:, inputs.input_ids.shape[1]:]

        result_text = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": req.prompt,
            "response": result_text,
            "meta": {
                "max_new_tokens": MAX_NEW_TOKENS
            }
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return InferResponse(text=result_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
