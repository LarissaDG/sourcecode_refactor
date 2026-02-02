import pandas as pd
from src.utils.file_operations import unzip_file, load_csv, save_csv
from sourcecode_refactor.src.preprocessing.columns import initialize_columns
from model_operations import load_model, process_images

#TODO implementar o pipeline e apagar

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
