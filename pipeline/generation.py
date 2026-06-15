import os
import torch
from PIL import Image


# ── Carregamento do modelo ────────────────────────────────────────────────────

def _load_janus(model_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM      # lazy import
    from janus.models import VLChatProcessor           # lazy import
    processor = VLChatProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).to(device).eval()
    return model, processor


# ── Geração de uma imagem ─────────────────────────────────────────────────────

def _generate_image(
    caption: str,
    model,
    processor,
    device: torch.device,
    num_images: int = 1,
) -> list[Image.Image]:
    """
    Gera imagens a partir de uma caption usando o modo text-to-image do Janus-Pro.
    Retorna lista de PIL Images.
    """
    conversation = [
        {
            "role": "<|User|>",
            "content": caption,
        },
        {
            "role": "<|Assistant|>",
            "content": "",
        },
    ]

    inputs = processor(
        conversations=conversation,
        force_batchify=True,
    ).to(device)

    embeds = model.prepare_inputs_embeds(**inputs)

    outputs = model.generate(
        inputs_embeds=embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=576,    # 24x24 tokens de imagem para Janus
        num_return_sequences=num_images,
        do_sample=True,
        use_cache=True,
    )

    # Decodifica tokens de imagem → PIL
    images = []
    for out in outputs:
        img = processor.decode_image(out)
        if img is not None:
            images.append(img)
    return images


# ── Salva resultados de um modelo ─────────────────────────────────────────────

def _run_single_model(
    model_cfg: dict,
    data: list,
    base_output_dir: str,
    num_images: int,
    device: torch.device,
) -> list:
    """
    Roda geração para UM modelo (ex: Janus-Pro-1B ou Janus-Pro-7B).
    Salva imagens em: <output_dir>/generated/<model_name>/<filename>.png
    Retorna `data` enriquecido com o campo 'generated_<model_name>'.
    """
    model_name = model_cfg["name"]
    model_path = model_cfg["model_path"]
    save_dir   = os.path.join(base_output_dir, "generated", model_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[generation] Carregando {model_name}...")
    model, processor = _load_janus(model_path, device)

    for sample in data:
        caption   = sample.get("caption")
        filename  = os.path.splitext(sample["filename"])[0]  # sem extensão

        if not caption:
            print(f"  [!] Sem caption para {sample['filename']} — pulando.")
            sample[f"generated_{model_name}"] = []
            continue

        try:
            images = _generate_image(caption, model, processor, device, num_images)
        except Exception as e:
            print(f"  [!] Erro ao gerar para {sample['filename']}: {e}")
            sample[f"generated_{model_name}"] = []
            continue

        saved_paths = []
        for i, img in enumerate(images):
            suffix = f"_{i}" if num_images > 1 else ""
            out_path = os.path.join(save_dir, f"{filename}{suffix}.png")
            img.save(out_path)
            saved_paths.append(out_path)

        sample[f"generated_{model_name}"] = saved_paths
        print(f"  ✓ {sample['filename']} → {len(saved_paths)} imagem(ns) salva(s)")

    # Libera VRAM antes de carregar o próximo modelo
    del model
    torch.cuda.empty_cache()

    return data


# ── Caixinha 3 ────────────────────────────────────────────────────────────────

def run_generation(cfg, data: list) -> list:
    """
    Caixinha 3 — Geração de imagens com Janus-Pro.

    Para cada modelo listado em cfg['generation']['models']:
      - Carrega o modelo
      - Gera imagens a partir das captions em `data`
      - Salva em outputs/<exp_name>/generated/<model_name>/
      - Descarrega o modelo da VRAM antes de carregar o próximo

    Retorna `data` com campos 'generated_<model_name>' adicionados.
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_cfg = cfg["generation"]["models"]         # lista [{name, model_path}, ...]
    num_images = cfg["generation"].get("num_images_per_prompt", 1)
    output_dir = os.path.join(cfg["experiment"]["output_dir"], cfg["experiment"]["name"])

    for model_cfg in models_cfg:
        data = _run_single_model(model_cfg, data, output_dir, num_images, device)

    return data