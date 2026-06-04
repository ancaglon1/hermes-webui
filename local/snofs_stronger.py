#!/usr/bin/env python3
"""Stronger SNOFS - more explicit prompts on v14Distilled."""

import json, time, urllib.request, urllib.error, sys, random

HOST = "http://3xs.osiris-eel.ts.net:8187"

prompts = [
    # Stronger explicit
    ("extremely explicit photorealistic image of a elven warrior woman in a palatial bedroom, partially naked, ornate glowing armor, perfect slim athletic body, small natural tits with erect nipples, trimmed pussy, glistening wet skin, long messy golden hair, huge glowing blue greatsword, seductive smirk, beckoning viewer, dramatic studio lighting, photorealistic, 8k, highly detailed, nude, explicit hardcore erotic art", ""),
    # Even stronger
    ("RAW porn photo of a naked elven warrior in a forest clearing during golden hour, completely nude, small tits, hard nipples, shaved wet pussy, toned slim body, sweaty glowing skin, long golden hair, smiling seductively at viewer, wide parted lips, spreading pussy with fingers, dramatic lighting, photorealistic extreme detail, explicit hardcore", "nsfw, blurred, censored, watermark"),
]

for i, (prompt, neg) in enumerate(prompts):
    seed = random.randint(0, 2**32 - 1)
    print(f"\n=== Run {i+1} (seed: {seed}) ===", flush=True)
    n = {}
    n["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "snofsSexNudesAndOtherFunStuff_v14Distilled.safetensors", "weight_dtype": "default"}}
    n["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2"}}
    n["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}
    n["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}}
    n["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}}
    n["6"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}}
    n["7"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": 4, "cfg": 3.5, "sampler_name": "euler", "scheduler": "sgm_uniform", "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "denoise": 1.0}}
    n["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    n["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": f"snofs_v14_run{i+1}"}}

    try:
        resp = urllib.request.urlopen(urllib.request.Request(f"{HOST}/prompt", data=json.dumps({"prompt": n}).encode(), headers={"Content-Type": "application/json"}, method="POST"), timeout=60)
        pid = json.loads(resp.read())["prompt_id"]
        print(f"OK: {pid}", flush=True)
        for t in range(60):
            time.sleep(5)
            try:
                hist = json.loads(urllib.request.urlopen(f"{HOST}/history/{pid}", timeout=15).read())
                if pid in hist:
                    s = hist[pid].get("status",{}).get("status_str","?")
                    imgs = [img for nid, no in hist[pid].get("outputs",{}).items() if "images" in no for img in no["images"]]
                    if s == "success" and imgs:
                        img = imgs[0]
                        url = f"{HOST}/view?filename={img['filename']}&type={img.get('type','output')}"
                        dl = urllib.request.urlopen(url, timeout=60).read()
                        path = f"/home/hermes/hermes-webui/{img['filename']}"
                        with open(path, "wb") as f: f.write(dl)
                        print(f"SAVED: {path} (seed: {seed})", flush=True)
                    elif s == "error":
                        for m in hist[pid].get("status",{}).get("messages",[]):
                            if m[0] == "execution_error":
                                print(f"ERR: {m[1].get('exception_message','')[:200]}", flush=True)
                    if s in ("success", "error"): break
            except: pass
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:200]}", flush=True)
    except Exception as e:
        print(f"FAIL: {e}", flush=True)
