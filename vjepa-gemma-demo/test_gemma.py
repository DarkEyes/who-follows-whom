"""Probe llama-server: text-only call, then one-image call.
Prints raw responses so the failure layer is visible."""

import base64
import json

import numpy as np
import cv2
import requests

URL = "http://localhost:8080/v1/chat/completions"


def ask(messages, tag):
    r = requests.post(URL, json={
        "model": "gemma",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.1,
    }, timeout=300)
    print(f"--- {tag}: HTTP {r.status_code}")
    try:
        j = r.json()
        print(json.dumps(j, indent=2)[:1500])
        content = j["choices"][0]["message"]["content"]
        print(f"--- {tag} content repr: {content!r}")
    except Exception as e:
        print("parse failure:", e, r.text[:500])


ask([{"role": "user", "content": "Reply with exactly: hello world"}],
    "TEXT-ONLY")

img = np.zeros((256, 256, 3), np.uint8)
cv2.circle(img, (128, 128), 60, (0, 0, 255), -1)
ok, buf = cv2.imencode(".jpg", img)
b64 = base64.b64encode(buf.tobytes()).decode()
ask([{"role": "user", "content": [
    {"type": "text", "text": "What shape and color do you see?"},
    {"type": "image_url",
     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
]}], "ONE-IMAGE")
