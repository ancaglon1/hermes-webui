#!/usr/bin/env python3
"""Generate explicit elven warrior with all 4 Flux-2 Klein models + LoRA."""

import json, time, urllib.request, urllib.error, sys, os, random

HOST = "http://3xs.osiris-eel.ts.net:8187"

# Same base prompt for all, adapted for Flux
PROMPT_CLIP_L = (
    "RAW erotic photo of a naked elven warrior woman, "
    "completely nude, slim athletic body, perfect natural breasts, "
    "shaved pussy, wet glistening skin, short golden hair messy, "
    "on knees in battle stance holding a huge glowing blue sword, "
    "intense lustful determined gaze, parted lips, flushed skin, "
    "dramatic studio lighting, photorealistic, ultra detailed, 8k, "
    "nude, explicit, erotic art"
)
PROMPT_T5 = PROMPT_CLIP_L  # Same prompt for both encoders

NEGATIVE = (
    "blurry, low quality, distorted, ugly, bad anatomy, "
    "deformed, extra limbs, bad hands, bad face, malformed, "
    "watermark, signature, text, censored, mosaic, "
    "clothes, covered, armor"
)

# DualCLIP type needs to be "flux" for standard Flux models
CLIP_TYPE = "flux"
CLIP_L = "clip_l.safetensors"
T5XXL = "t5xxl_fp8_e4m3fn_scaled.safetensors"
VAE_NAME = "ae.safetensors"
LORA_NAME = "Flux/klein_snofs_v1_3.safetensors"

MODELS = [
    "flux-2-klein-4b.safetensors",
    "flux-2-klein-9b-kv-fp8.safetensors",
    "flux-2-klein-base-4b.safetensors",
    "flux-2-klein-base-9b-fp8.safetensors",
]

RESULTS = []

for model_name in MODELS:
    seed = random.randint(0, 2**32 - 1)
    print(f"\n=== Trying: {model_name} (seed: {seed}) ===", flush=True)

    # Build node IDs sequentially so they don't conflict across runs
    n = {}
    n["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": model_name, "weight_dtype": "default"}}
    n["2"] = {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": CLIP_L, "clip_name2": T5XXL, "type": CLIP_TYPE}}
    n["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}}
    n["4"] = {"class_type": "ModelSamplingFlux", "inputs": {"model": ["1", 0], "max_shift": 1.15, "base_shift": 0.5, "width": 1024, "height": 1024}}
    n["5"] = {"class_type": "LoraLoader", "inputs": {"model": ["4", 0], "clip": ["2", 0], "lora_name": LORA_NAME, "strength_model": 0.8, "strength_clip": 0.8}}
    n["6"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip": ["5", 1], "clip_l": PROMPT_CLIP_L, "t5xxl": PROMPT_T5, "guidance": 3.0}}
    n["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["5", 1]}}
    n["8"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": 3.0}}
    n["9"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024}}
    n["10"] = {"class_type": "KSampler", "inputs": {
        "model": ["5", 0],
        "seed": seed,
        "steps": 25,
        "cfg": 3.0,
        "sampler_name": "euler",
        "scheduler": "sgm_uniform",
        "positive": ["8", 0],
        "negative": ["7", 0],
        "latent_image": ["9", 0],
        "denoise": 1.0
    }}
    n["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
    prefix = f"flux2_{model_name.replace('.safetensors','').replace('flux-2-','')}"
    n["12"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}}

    # Submit
    payload = json.dumps({"prompt": n}).encode()
    req = urllib.request.Request(f"{HOST}/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        prompt_id = result["prompt_id"]
        print(f"Submitted: {prompt_id}", flush=True)
    except Exception as e:
        print(f"Submit failed: {e}", flush=True)
        RESULTS.append((model_name, None, f"submit failed: {e}"))
        continue

    # Poll
    success = False
    for i in range(120):
        time.sleep(3)
        try:
            hist = json.loads(urllib.request.urlopen(f"{HOST}/history/{prompt_id}", timeout=15).read())
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {}).get("status_str", "unknown")
                output_data = hist[prompt_id].get("outputs", {})
                if status == "success":
                    for node_id, node_out in output_data.items():
                        if "images" in node_out:
                            img = node_out["images"][0]
                            fn, sf, tp = img["filename"], img.get("subfolder",""), img.get("type","output")
                            url = f"{HOST}/view?filename={fn}&type={tp}" + (f"&subfolder={sf}" if sf else "")
                            dl = urllib.request.urlopen(url, timeout=60).read()
                            path = f"/home/hermes/hermes-webui/{fn}"
                            with open(path, "wb") as f: f.write(dl)
                            print(f"SAVED: {path}", flush=True)
                            RESULTS.append((model_name, path, f"seed={seed}"))
                            success = True
                            break
                    if not success:
                        print(f"Success but no images: {json.dumps(output_data)[:200]}", flush=True)
                        RESULTS.append((model_name, None, "no images in output"))
                else:
                    # Check for errors
                    error_msg = hist[prompt_id].get("status", {}).get("messages", [])
                    print(f"Failed: {json.dumps(error_msg)[:300]}", flush=True)
                    RESULTS.append((model_name, None, f"status: {status}"))
                break
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"HTTP {e.code}")
                if i > 10:
                    RESULTS.append((model_name, None, f"HTTP {e.code}"))
                    break
            if i % 20 == 0: print(f"  Waiting ({i*3}s)", flush=True)
        except Exception as e:
            if i % 20 == 0: print(f"  Retry: {e}", flush=True)
            time.sleep(5)

    if not success and not RESULTS or RESULTS[-1][0] != model_name:
        RESULTS.append((model_name, None, "timeout"))

print("\n\n=== RESULTS ===", flush=True)
for m, path, note in RESULTS:
    status = f"SAVED: {path}" if path else f"FAILED: {note}"
    print(f"  {m}: {status}", flush=True)
