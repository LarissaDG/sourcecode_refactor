import pandas as pd
import zipfile
import os
from pathlib import Path
import tempfile
from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


# -------------------------------
# 1) DESCOMPACTAR ZIP
# -------------------------------
def unzip_file(zip_path: str, unzip_dir: str):
    print("Descompactando o arquivo ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print("Descompactação concluída.")


# -------------------------------
# 2) CARREGAR CSV
# -------------------------------
def load_csv(csv_path: str) -> pd.DataFrame:
    print(f"Carregando o arquivo CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print("Arquivo CSV carregado com sucesso.")
    return df


# -------------------------------
# 3) INICIALIZAR NOVAS COLUNAS
# -------------------------------
def initialize_columns(df: pd.DataFrame):
    new_columns = [
        'Total aesthetic score', 'Theme and logic', 'Creativity', 'Layout and composition',
        'Space and perspective', 'The sense of order', 'Light and shadow', 'Color',
        'Details and texture', 'The overall', 'Mood', 'generated_filename', 'Description'
    ]
    for col in new_columns:
        df[col] = ''  # Inicializando com valores vazios
    return df


# -------------------------------
# 4) CARREGAR MODELO
# -------------------------------
def load_model(model_path: str):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    return vl_chat_processor, tokenizer, vl_gpt


# -------------------------------
# 5) GERAR DESCRIÇÃO DA IMAGEM
# -------------------------------
def generate_description(image_path: str, vl_chat_processor, tokenizer, vl_gpt):
    try:
        print(f"Processando a imagem: {image_path}")
        image = Image.open(image_path)

        # Salvar a imagem em um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file, format="PNG")
            temp_image_path = temp_file.name

        # Preparar a descrição
        question = "Describe this image."
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{question}", "images": [temp_image_path]},
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # Rodar o modelo para gerar a resposta
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        # Decodificar a resposta
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"Descrição gerada para {image_path}: {answer[:100]}...")
        return answer

    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")
        return None


# -------------------------------
# 6) PROCESSAR IMAGENS E ATUALIZAR CSV
# -------------------------------
def process_images(df: pd.DataFrame, image_base_path: str, vl_chat_processor, tokenizer, vl_gpt):
    print("Iniciando o processamento das imagens...")
    for idx, row in df.iterrows():
        image_path = os.path.join(image_base_path, row['filename'])
        description = generate_description(image_path, vl_chat_processor, tokenizer, vl_gpt)
        df.at[idx, 'Description'] = description
        print(f"Descrição adicionada para {row['filename']}.")
    return df


# -------------------------------
# 7) SALVAR CSV
# -------------------------------
def save_csv(df: pd.DataFrame, output_csv_path: str):
    df.to_csv(output_csv_path, index=False)
    print(f"Arquivo CSV atualizado salvo em {output_csv_path}")


# -------------------------------
# 8) EXECUÇÃO PRINCIPAL
# -------------------------------
def main():
    # Caminhos
    zip_path = "/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_images.zip"
    unzip_dir = "/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_images/"
    csv_path = "/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_dataset.csv"
    image_base_path = "/home_cerberus/disk3/larissa.gomide/APDDv2/APDDv2images/"
    output_csv_path = "/home_cerberus/disk3/larissa.gomide/oficial/sampled_dataset_with_descriptions.csv"

    # Pipeline
    unzip_file(zip_path, unzip_dir)
    df = load_csv(csv_path)
    df = initialize_columns(df)

    vl_chat_processor, tokenizer, vl_gpt = load_model("deepseek-ai/Janus-Pro-7B")
    df = process_images(df, image_base_path, vl_chat_processor, tokenizer, vl_gpt)

    save_csv(df, output_csv_path)


if __name__ == "__main__":
    main()
