#!/usr/bin/env python3
"""Generate explicit image using Z-Image Turbo NSFW fine-tune."""

import json, time, urllib.request, urllib.error, sys, os, random

HOST = "http://3xs.osiris-eel.ts.net:8187"
SEED = random.randint(0, 2**32 - 1)

PROMPT = (
    "explicit erotic RAW photo of a naked elven warrior woman, "
    "completely nude, slim athletic body, perfect breasts with hard nipples, "
    "shaved pussy, glistening wet skin, short golden hair messy, "
    "kneeling in battle stance holding a huge glowing blue sword, "
    "intense lustful look, parted lips, flushed, "
    "dramatic studio lighting, photorealistic, ultra detailed, 8k, "
    "nude, explicit, erotic, porn"
)
NEGATIVE = (
    "blurry, low quality, distorted, ugly, bad anatomy, "
    "deformed, extra limbs, bad hands, bad face, "
    "watermark, signature, text, censored, mosaic, "
    "clothes, covered, armor"
)

workflow = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "zImageturboVISIONARYNSFW_zitFp8V01.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "omnigen2"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "4": {"class_type": "TextEncodeZImageOmni", "inputs": {"clip": ["2", 0], "prompt": PROMPT, "auto_resize_images": True}},
    "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}},
    "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 1024, "batch_size": 1}},
    "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "8": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0}},
    "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "10": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["4", 0]}},
    "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["10", 0], "sampler": ["7", 0], "sigmas": ["8", 0], "latent_image": ["6", 0]}},
    "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
    "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "zit_nsfw_elf"}}
}

payload = json.dumps({"prompt": workflow}).encode()
req = urllib.request.Request(f"{HOST}/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST")
resp = urllib.request.urlopen(req, timeout=30)
result = json.loads(resp.read())
prompt_id = result["prompt_id"]
print(f"Submitted: {prompt_id}", flush=True)

for i in range(60):
    time.sleep(3)
    try:
        hist = json.loads(urllib.request.urlopen(f"{HOST}/history/{prompt_id}", timeout=10).read())
        if prompt_id in hist:
            output_data = hist[prompt_id].get("outputs", {})
            for node_id, node_out in output_data.items():
                if "images" in node_out:
                    img = node_out["images"][0]
                    fn, sf, tp = img["filename"], img.get("subfolder",""), img.get("type","output")
                    url = f"{HOST}/view?filename={fn}&type={tp}" + (f"&subfolder={sf}" if sf else "")
                    dl = urllib.request.urlopen(url, timeout=60).read()
                    path = f"/home/hermes/hermes-webui/{fn}"
                    with open(path, "wb") as f: f.write(dl)
                    print(f"SAVED: {path}", flush=True)
                    print(f"SEED: {SEED}", flush=True)
                    sys.exit(0)
            print(f"No images")
            sys.exit(1)
    except urllib.error.HTTPError as e:
        if e.code != 404: print(f"HTTP {e.code}")
        if i % 10 == 0: print(f"Waiting ({i*3}s)")
    except Exception as e:
        if i % 10 == 0: print(f"Retry")
        time.sleep(5)
print("TIMEOUT")
sys.exit(1)
