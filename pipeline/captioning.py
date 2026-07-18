import tempfile
import torch
from PIL import Image
from torch.utils.data import DataLoader


def _load_janus(model_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM          # lazy import
    from janus.models import VLChatProcessor               # lazy import
    global load_pil_images
    from janus.utils.io import load_pil_images              # lazy import
    processor = VLChatProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).to(device).eval()
    return model, processor


def _describe_image(image: Image.Image, model, processor, device) -> str | None:
    """Gera uma descrição para uma única imagem PIL usando Janus-Pro."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name

        conversation = [
            {"role": "<|User|>", "content": "<image_placeholder>\nDescribe this image.", "images": [tmp_path]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_imgs = load_pil_images(conversation)
        inputs = processor(conversations=conversation, images=pil_imgs, force_batchify=True).to(device)
        embeds = model.prepare_inputs_embeds(**inputs)
        out = model.language_model.generate(
            inputs_embeds=embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        return processor.tokenizer.decode(out[0].cpu().tolist(), skip_special_tokens=True)
    except Exception as e:
        print(f"Erro ao gerar descrição: {e}")
        return None


def run_captioning(cfg, loader: DataLoader) -> list:
    """
    Caixinha 2 — Geração de descrições com Janus-Pro.
    Recebe DataLoader, devolve lista de dicts com 'filename', 'image', 'caption'.
    Se o dataset já tiver caption (ex: Portinari com use_human_captions),
    o campo 'caption' já vem preenchido e esta caixinha é pulada pelo pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = cfg["captioning"].get("model_path") or "deepseek-ai/Janus-Pro-7B"
    model, processor = _load_janus(model_path, device)

    results = []
    for batch in loader:
        for i, filename in enumerate(batch["filename"]):
            img_path = batch["path"][i]
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception:
                print(f"  [!] Imagem não encontrada, pulando captioning: {img_path}")
                continue
            caption = _describe_image(pil_img, model, processor, device)
            results.append({
                "filename": filename,
                "path":     img_path,
                "image":    batch["image"][i],
                "caption":  caption,
            })
    return results