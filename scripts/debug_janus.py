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

caption = "A red apple on a wooden table."

# Prompt SFT + image_start_tag (igual ao script original get_gen_img.py)
conversation = [
    {"role": "User",      "content": caption},
    {"role": "Assistant", "content": ""},
]
sft = processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=processor.sft_format,
    system_prompt="",
)
prompt = sft + processor.image_start_tag
print("Prompt:", repr(prompt[:120]))

input_ids = processor.tokenizer.encode(prompt)
input_ids = torch.LongTensor(input_ids)

IMAGE_TOKEN_NUM = 576
IMG_SIZE        = 384
PATCH_SIZE      = 16
CFG_WEIGHT      = 5.0
TEMPERATURE     = 1.0

tokens = torch.zeros((2, len(input_ids)), dtype=torch.int).to(device)
tokens[0, :] = input_ids
tokens[1, :] = input_ids
tokens[1, 1:-1] = processor.pad_id

inputs_embeds = model.language_model.get_input_embeddings()(tokens)
generated_tokens = torch.zeros((1, IMAGE_TOKEN_NUM), dtype=torch.int).to(device)
past_key_values = None

print("Gerando tokens de imagem...")
with torch.inference_mode():
    for i in range(IMAGE_TOKEN_NUM):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        logits = model.gen_head(outputs.last_hidden_state[:, -1, :])
        logit_cond, logit_uncond = logits[0:1], logits[1:2]
        logits = logit_uncond + CFG_WEIGHT * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(-1)
        next_token_pair = next_token.repeat(2, 1).view(-1)
        inputs_embeds = model.prepare_gen_img_embeds(next_token_pair).unsqueeze(1)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{IMAGE_TOKEN_NUM} tokens gerados")

import numpy as np
dec = model.gen_vision_model.decode_code(
    generated_tokens.to(torch.int),
    shape=[1, 8, IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE],
)
dec = dec.to(torch.float32).cpu().detach().numpy().transpose(0, 2, 3, 1)
dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

import PIL.Image
img = PIL.Image.fromarray(dec[0])
out = "/tmp/debug_janus_output.png"
img.save(out)
print(f"Imagem salva em: {out} ({img.size})")
