import os
os.environ.setdefault("HF_HOME",            "/snfs1/speed/larissa.gomide/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/snfs1/speed/larissa.gomide/hf_cache")
os.environ.setdefault("CLIP_CACHE",         "/snfs1/speed/larissa.gomide/hf_cache")

import torch
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

model_path = "deepseek-ai/Janus-Pro-1B"
print(f"Carregando {model_path}...")
processor = VLChatProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model = model.to(torch.bfloat16).to(device).eval()
print("Modelo carregado.")

conversation = [
    {"role": "<|User|>", "content": "A red apple on a wooden table."},
    {"role": "<|Assistant|>", "content": ""},
]
inputs = processor(conversations=conversation, force_batchify=True).to(device)
embeds = model.prepare_inputs_embeds(**inputs)

print("Gerando...")
outputs = model.generate(
    inputs_embeds=embeds,
    attention_mask=inputs.attention_mask,
    pad_token_id=processor.tokenizer.eos_token_id,
    bos_token_id=processor.tokenizer.bos_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    max_new_tokens=576,
    do_sample=True,
    use_cache=True,
)
print("Output shape:", outputs.shape)
print("Primeiros tokens:", outputs[0][:20].tolist())

for i, out in enumerate(outputs):
    img = processor.decode_image(out)
    print(f"Image {i}: {img}")
    if img is not None:
        img.save(f"/tmp/debug_janus_{i}.png")
        print(f"  Salva em /tmp/debug_janus_{i}.png")
