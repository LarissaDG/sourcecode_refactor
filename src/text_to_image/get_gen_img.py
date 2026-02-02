import argparse
from tex2img_operations import run_model_generation
from src.utils.file_operations import load_csv

#TODO implementar o pipeline e apagar

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
