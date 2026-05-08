from src.utils.file_operations import unzip_file, load_csv
from src.preprocessing.columns import add_score_columns, initialize_columns
from src.sampling.sampling import Sampler

def run_preprocessing(self):
        #Ler .yaml
        # Aqui você pega toda a seção preprocessing.
        cfg = self.config["steps"]["preprocessing"]
        #Depois você acessa os parâmetros:
        #zip_path = cfg["params"]["zip_path"]

        #Abrir csv
        csv_path = cfg["params"]["csv_path"]

        self.df = load_csv(csv_path)

        cols_to_compare = self.config["cols_to_compare"]

        #processamento
        self.df = add_score_columns(self.df, cols_to_compare)
        # inicializar colunas
        self.df = initialize_columns(self.df,cols_to_compare)
       
def run_sampling(self):
    cfg = self.config["steps"]["sampling"]["params"]

    sampler = Sampler(
        n_samples=cfg["n_samples"],
        n_bins=cfg.get("n_bins", 30),
        seed=cfg.get("seed", 42),
        method=cfg.get("method", "both"),
        score_column=cfg.get("score_column", "Avg Score")
    )

    results = sampler.run(self.df)

    # Decide o que vira self.df
    # Exemplo: usar uniforme como default
    if "uniform" in results:
        self.df = results["uniform"]
    else:
        self.df = next(iter(results.values()))

def run_image_to_text(self):
        cfg = self.config["steps"]["image_to_text"]

        model_path = cfg["params"]["model_path"]
        image_base_path = cfg["params"]["image_base_path"]

        self.vl_chat_processor, self.tokenizer, self.vl_gpt = load_model(model_path)
        self.df = process_images(self.df, image_base_path, self.vl_chat_processor, self.tokenizer, self.vl_gpt)

def run_text_to_image(self):
        # Se não existir ainda, deixa como placeholder
        self.logger.info("Text-to-image ainda não implementado")

def run_scoring(self):
        cfg = self.config["steps"]["scoring"]
        cols = cfg["params"]["cols"]

        self.df = add_score_columns(self.df, cols)

def run_iccc_metrics(self):