#!/usr/bin/env python3
"""SNOFS via exact Reddit workflow: LoraLoaderModelOnly + CFGGuider + Flux2Scheduler."""

import json, time, urllib.request, urllib.error, sys, os, random

HOST = "http://3xs.osiris-eel.ts.net:8187"
SEED = random.randint(0, 2**32 - 1)

PROMPT = "nude elven warrior woman, naked, slim athletic body, perfect breasts, shaved pussy, wet glistening skin, short golden hair messy, huge glowing blue sword, battle stance, determined lustful gaze, dramatic lighting, photorealistic, detailed, erotic, explicit"
NEGATIVE = ""

W = 832
H = 1216

n = {}
n["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-base-9b-fp8.safetensors", "weight_dtype": "default"}}
n["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2"}}
n["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
n["4"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Flux/klein_snofs_v1_3.safetensors", "strength_model": 1.0}}
n["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}}
n["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}}
n["7"] = {"class_type": "CFGGuider", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "cfg": 5}}
n["8"] = {"class_type": "Flux2Scheduler", "inputs": {"steps": 50, "width": W, "height": H}}
n["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
n["10"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
n["11"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}}
n["12"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["7", 0], "sampler": ["9", 0], "sigmas": ["8", 0], "latent_image": ["11", 0]}}
n["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}}
n["14"] = {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "snofs_reddit"}}

payload = json.dumps({"prompt": n}).encode()
req = urllib.request.Request(f"{HOST}/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST")
try:
    resp = urllib.request.urlopen(req, timeout=30)
    pid = json.loads(resp.read())["prompt_id"]
    print(f"OK: {pid}", flush=True)
except Exception as e:
    print(f"SUBMIT FAIL: {e}", flush=True)
    if hasattr(e, 'read'):
        print(e.read().decode()[:500], flush=True)
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
                dl = urllib.request.urlopen(url, timeout=120).read()
                path = f"/home/hermes/hermes-webui/{fn}"
                with open(path, "wb") as f: f.write(dl)
                print(f"SAVED: {path} (seed: {SEED})", flush=True)
                break
            elif s == "error":
                for m in hist[pid].get("status",{}).get("messages",[]):
                    if m[0] == "execution_error":
                        print(f"ERROR: {m[1].get('exception_message','')[:300]}", flush=True)
            if s in ("success", "error"): break
        if i % 10 == 0: print(f"wait ({i*5}s)", flush=True)
    except: time.sleep(5)
