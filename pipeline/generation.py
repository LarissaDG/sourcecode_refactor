import os
import numpy as np
import torch
import PIL.Image


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(processor, caption: str, use_alternate: bool) -> str:
    """
    Constrói o prompt SFT para geração de imagem.
    1B usa roles "<|User|>"/"<|Assistant|>" (use_alternate=False não é o certo —
    na verdade 1B usa User/Assistant sem pipes, 7B usa com pipes).
    """
    if use_alternate:
        conversation = [
            {"role": "<|User|>", "content": caption},
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        conversation = [
            {"role": "User", "content": caption},
            {"role": "Assistant", "content": ""},
        ]

    sft = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    return sft + processor.image_start_tag


# ── Geração token-a-token com CFG ────────────────────────────────────────────

@torch.inference_mode()
def _generate_token_based(
    model,
    processor,
    prompt: str,
    device: torch.device,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    image_token_num: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
) -> PIL.Image.Image:
    """
    Gera UMA imagem via loop token-a-token com classifier-free guidance.
    Baseado no script original get_gen_img.py.
    """
    input_ids = processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    # Duas cópias: condicional (par) e incondicional (ímpar)
    tokens = torch.zeros((2, len(input_ids)), dtype=torch.int).to(device)
    tokens[0, :] = input_ids
    tokens[1, :] = input_ids
    tokens[1, 1:-1] = processor.pad_id  # mascara o prompt na cópia incondicional

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int).to(device)
    past_key_values = None

    for i in range(image_token_num):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond   = logits[0:1, :]
        logit_uncond = logits[1:2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens[:, i] = next_token.squeeze(-1)

        next_token_pair = next_token.repeat(2, 1).view(-1)
        img_embeds = model.prepare_gen_img_embeds(next_token_pair)
        inputs_embeds = img_embeds.unsqueeze(1)

    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(torch.int),
        shape=[1, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().detach().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(dec[0])


# ── Carregamento do modelo ────────────────────────────────────────────────────

def _load_janus(model_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM
    from janus.models import VLChatProcessor
    processor = VLChatProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).to(device).eval()
    return model, processor


# ── Salva resultados de um modelo ─────────────────────────────────────────────

def _run_single_model(
    model_cfg: dict,
    data: list,
    base_output_dir: str,
    num_images: int,
    device: torch.device,
) -> list:
    model_name   = model_cfg["name"]
    model_path   = model_cfg["model_path"]
    use_alternate = "7B" in model_name  # 7B usa roles com pipes, 1B sem
    save_dir     = os.path.join(base_output_dir, "generated", model_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[generation] Carregando {model_name}...")
    model, processor = _load_janus(model_path, device)

    for sample in data:
        caption  = sample.get("caption", "")
        filename = os.path.splitext(sample["filename"])[0]

        if not caption:
            print(f"  [!] Sem caption para {sample['filename']} — pulando.")
            sample[f"generated_{model_name}"] = []
            continue

        prompt = _build_prompt(processor, caption, use_alternate=use_alternate)
        saved_paths = []

        for i in range(num_images):
            try:
                img = _generate_token_based(model, processor, prompt, device)
                suffix   = f"_{i}" if num_images > 1 else ""
                out_path = os.path.join(save_dir, f"{filename}{suffix}.png")
                img.save(out_path)
                saved_paths.append(out_path)
            except Exception as e:
                print(f"  [!] Erro ao gerar imagem {i} para {sample['filename']}: {e}")

        sample[f"generated_{model_name}"] = saved_paths
        print(f"  ✓ {sample['filename']} → {len(saved_paths)} imagem(ns)")

    del model
    torch.cuda.empty_cache()
    return data


# ── Caixinha 3 ────────────────────────────────────────────────────────────────

def run_generation(cfg, data: list) -> list:
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_cfg = cfg["generation"]["models"]
    num_images = cfg["generation"].get("num_images_per_prompt", 1)
    output_dir = os.path.join(cfg["experiment"]["output_dir"], cfg["experiment"]["name"])

    for model_cfg in models_cfg:
        data = _run_single_model(model_cfg, data, output_dir, num_images, device)

    return data
