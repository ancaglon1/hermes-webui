#!/usr/bin/env python3
"""Generate with 9B Klein models + snofs LoKr via LycorisLoader (proper LoKr support)."""

import json, time, urllib.request, urllib.error, sys, os, random

HOST = "http://3xs.osiris-eel.ts.net:8187"

PROMPT = "nude elven warrior woman, naked, slim athletic body, perfect breasts, shaved pussy, wet glistening skin, short golden hair messy, huge glowing blue sword, battle stance, determined lustful gaze, dramatic lighting, photorealistic, detailed, erotic, explicit"
NEGATIVE = "blurry, low quality, distorted, ugly, bad anatomy, deformed, extra limbs, bad hands, watermark, text, censored"

VAE = "flux2-vae.safetensors"
CLIP = "qwen_3_8b_fp8mixed.safetensors"
LORA = "Flux/klein_snofs_v1_3.safetensors"

MODELS = [
    ("flux-2-klein-base-9b-fp8.safetensors", "base9b"),
    ("flux-2-klein-9b-kv-fp8.safetensors", "9b_kv"),
]

for model_name, label in MODELS:
    seed = random.randint(0, 2**32 - 1)
    print(f"\n=== {label}: {model_name} ===", flush=True)

    n = {}
    n["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": model_name, "weight_dtype": "default"}}
    n["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "flux2"}}
    n["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}
    # Use LycorisLoaderNode instead of LoraLoader for proper LoKr support
    n["4"] = {"class_type": "LycorisLoaderNode", "inputs": {
        "model": ["1", 0], "clip": ["2", 0],
        "lora_name": LORA, "strength_model": 0.8, "strength_clip": 0.8
    }}
    n["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["4", 1]}}
    n["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["4", 1]}}
    n["7"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    n["8"] = {"class_type": "KSampler", "inputs": {
        "model": ["4", 0], "seed": seed, "steps": 25, "cfg": 3.5,
        "sampler_name": "euler", "scheduler": "sgm_uniform",
        "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "denoise": 1.0
    }}
    n["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}}
    n["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": f"flux2_{label}_snofs_lyco"}}

    payload = json.dumps({"prompt": n}).encode()
    req = urllib.request.Request(f"{HOST}/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)
    pid = json.loads(resp.read())["prompt_id"]
    print(f"Submitted: {pid}", flush=True)

    for i in range(120):
        time.sleep(5)
        try:
            hist = json.loads(urllib.request.urlopen(f"{HOST}/history/{pid}", timeout=15).read())
            if pid in hist:
                s = hist[pid].get("status",{}).get("status_str","?")
                imgs = []
                for nid, no in hist[pid].get("outputs",{}).items():
                    if "images" in no: imgs.extend(no["images"])
                if s == "success" and imgs:
                    img = imgs[0]
                    fn, sf, tp = img["filename"], img.get("subfolder",""), img.get("type","output")
                    url = f"{HOST}/view?filename={fn}&type={tp}" + (f"&subfolder={sf}" if sf else "")
                    dl = urllib.request.urlopen(url, timeout=120).read()
                    path = f"/home/hermes/hermes-webui/{fn}"
                    with open(path, "wb") as f: f.write(dl)
                    print(f"SAVED: {path} (seed: {seed})", flush=True)
                    break
                elif s == "error":
                    for m in hist[pid].get("status",{}).get("messages",[]):
                        if m[0] == "execution_error":
                            print(f"ERROR: {m[1].get('exception_message','')[:300]}", flush=True)
                if s in ("success", "error"):
                    break
        except Exception as e:
            if i % 10 == 0: print(f"waiting ({i*5}s)", flush=True)
            time.sleep(5)
    else:
        print("TIMEOUT", flush=True)
