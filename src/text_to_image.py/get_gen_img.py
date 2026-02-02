import os
import time
import torch
import numpy as np
import PIL.Image
import torchvision
import pandas as pd
import argparse

from transformers import AutoModelForCausalLM, AutoModel
from janus.models import MultiModalityCausalLM, VLChatProcessor


# -------------------------------
# 1) FUNÇÃO: CONSTRUIR PROMPT
# -------------------------------
def get_prompt(vl_chat_processor, prompt_text, is_token_based=True, use_alternate=False):
    if use_alternate:
        conversation = [
            {"role": "<|User|>", "content": prompt_text},
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        conversation = [
            {"role": "User", "content": prompt_text},
            {"role": "Assistant", "content": ""},
        ]

    print("DEBUG prompt:", prompt_text, type(prompt_text))

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + (vl_chat_processor.image_start_tag if is_token_based else vl_chat_processor.image_gen_tag)


# -------------------------------
# 2) FUNÇÃO: GERAR IMAGEM TOKEN-BASED
# -------------------------------
@torch.inference_mode()
def generate_token_based(
    mmgpt,
    vl_chat_processor,
    prompt,
    temperature=1.0,
    parallel_size=1,
    cfg_weight=5,
    image_token_num=576,
    img_size=384,
    patch_size=16,
    save_path="generated_samples/img_token.jpg"
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int).cuda()
    past_key_values = None

    for i in range(image_token_num):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])

        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    image = PIL.Image.fromarray(dec[0])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


# -------------------------------
# 3) FUNÇÃO: CARREGAR CSV
# -------------------------------
def load_csv(input_csv: str):
    try:
        df = pd.read_csv(input_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding="latin1")
    return df


# -------------------------------
# 4) FUNÇÃO: RODAR MODELO (GENERAÇÃO)
# -------------------------------
def run_model_generation(
    model_path: str,
    df: pd.DataFrame,
    output_dir: str,
    use_alternate: bool,
    model_name_prefix: str
):
    print(f"Executando modelo {model_name_prefix} ({model_path})...")

    processor = VLChatProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()

    os.makedirs(output_dir, exist_ok=True)

    generated_filenames = []
    for idx, row in df.iterrows():
        prompt_text = row["Description_en"]
        prompt = get_prompt(processor, prompt_text, is_token_based=True, use_alternate=use_alternate)

        out_filename = os.path.join(output_dir, f"{model_name_prefix}_{idx}.jpg")
        generate_token_based(model, processor, prompt, save_path=out_filename)

        generated_filenames.append(out_filename)
        print(f"[{model_name_prefix}] Processado prompt {idx}")

    df_out = df.copy()
    df_out["generated_filename"] = generated_filenames

    output_csv = f"sampled_{model_name_prefix}_with_gen.csv"
    df_out.to_csv(output_csv, index=False)


# -------------------------------
# 5) MAIN
# -------------------------------
def main(single=False):
    input_csv = "/sonic_home/larissa.gomide/TradBasePortinari/MiniBasePortinari_Translated.csv"

    df = load_csv(input_csv)

    # Se single=True, reduz o dataframe para apenas primeira linha
    if single:
        df = df.iloc[[0]]
        print("⚠️ Rodando APENAS com a primeira instância do CSV!")

    # Executar SMALL
    run_model_generation(
        model_path="deepseek-ai/Janus-Pro-1B",
        df=df,
        output_dir="/sonic_home/larissa.gomide/resultado_portinari/generated_oficial_small",
        use_alternate=False,
        model_name_prefix="img_small"
    )

    # Executar BIG
    run_model_generation(
        model_path="deepseek-ai/Janus-Pro-7B",
        df=df,
        output_dir="/sonic_home/larissa.gomide/resultado_portinari/generated_oficial_big",
        use_alternate=True,
        model_name_prefix="img_big"
    )


# -------------------------------
# PARSER DO ARGUMENTO
# -------------------------------
if __name__ == "__main__":
    print("SCRIPT INICIADO")

    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true",
                        help="Se usar, roda apenas a primeira linha do CSV")
    args = parser.parse_args()

    print("Roda main")
    main(single=args.single)
