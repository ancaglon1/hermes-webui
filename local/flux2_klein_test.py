#!/usr/bin/env python3
"""Test Flux 2 Klein 4B base with correct text encoder + LoRA."""

import json, time, urllib.request, urllib.error, sys, os, random

HOST = "http://3xs.osiris-eel.ts.net:8187"
SEED = random.randint(0, 2**32 - 1)

PROMPT = "nude elven warrior woman, naked, slim athletic body, perfect breasts, shaved pussy, wet skin, short golden hair, huge glowing blue sword, battle stance, determined lustful gaze, dramatic lighting, photorealistic, detailed, erotic, explicit"
NEGATIVE = "blurry, low quality, distorted, ugly, bad anatomy, deformed, extra limbs, bad hands, watermark, text, censored"

# Use model-appropriate text encoder
# 4B models → qwen_3_4b.safetensors
# 9B models → qwen_3_8b_fp8mixed.safetensors

CLIP_NAME = "qwen_3_4b.safetensors"   # for 4B models
VAE_NAME = "flux2-vae.safetensors"
LORA = "Flux/klein_snofs_v1_3.safetensors"
MODEL = "flux-2-klein-base-4b.safetensors"

n = {}
n["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}}
# Use single CLIPLoader with type="flux2" — this is the correct encoder for Flux 2 Klein
n["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "flux2"}}
n["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}}
# Apply LoRA
n["4"] = {"class_type": "LoraLoader", "inputs": {
    "model": ["1", 0], "clip": ["2", 0],
    "lora_name": LORA, "strength_model": 0.8, "strength_clip": 0.8
}}
# Positive prompt
n["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["4", 1]}}
# Negative prompt
n["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["4", 1]}}
# Latent image — Flux 2 Klein uses standard 16-channel latents
n["7"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
# KSampler with Flux-appropriate settings
n["8"] = {"class_type": "KSampler", "inputs": {
    "model": ["4", 0], "seed": SEED, "steps": 25, "cfg": 3.5,
    "sampler_name": "euler", "scheduler": "sgm_uniform",
    "positive": ["5", 0], "negative": ["6", 0],
    "latent_image": ["7", 0], "denoise": 1.0
}}
n["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}}
n["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "flux2_klein_test"}}

payload = json.dumps({"prompt": n}).encode()
req = urllib.request.Request(f"{HOST}/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST")
try:
    resp = urllib.request.urlopen(req, timeout=30)
    r = json.loads(resp.read())
    pid = r["prompt_id"]
    print(f"Submitted: {pid}", flush=True)
except urllib.error.HTTPError as e:
    print(f"Error {e.code}: {e.read().decode()[:500]}", flush=True)
    sys.exit(1)

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
                dl = urllib.request.urlopen(url, timeout=60).read()
                path = f"/home/hermes/hermes-webui/{fn}"
                with open(path, "wb") as f: f.write(dl)
                print(f"SAVED: {path}", flush=True)
                print(f"SEED: {SEED}", flush=True)
                sys.exit(0)
            elif s == "error":
                for m in hist[pid].get("status",{}).get("messages",[]):
                    if m[0] == "execution_error":
                        print(f"ERROR: {m[1].get('exception_message','')[:300]}", flush=True)
                sys.exit(1)
            else:
                print(f"Status: {s}, imgs: {len(imgs)}")
                if i > 60: break
        if i % 10 == 0: print(f"waiting ({i*5}s)", flush=True)
    except Exception as e:
        if i % 10 == 0: print(f"retry...", flush=True)
        time.sleep(5)
print("TIMEOUT")
sys.exit(1)
