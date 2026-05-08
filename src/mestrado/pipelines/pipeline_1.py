from src.pipelines.base_pipeline import BasePipeline

class Pipeline1(BasePipeline):
    
    #Pipeline do ICCC
    #Pipeline do Portinari -> Para deixar mais robusto repetir o pipeline colocando um condicional
    #Caso já tenha a coluna description. 
    #Colocar condicionais para caso já haja os arquivos não rodar todo o pipeline


    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.name = config["name"]

    def run_steps(self):
        self.logger.info(f"==== START PIPELINE: {self.name} ====")

        self.run_preprocessing()
        self.run_sampling()
        for method, df in self.sampled_dfs.items():
            self.logger.info(f"---- Running pipeline for sampling method: {method} ----")

            self.run_image_to_text()
            self.run_text_to_image()
            self.run_scoring()
            self.run_iccc_metrics()

        self.logger.info(f"==== END PIPELINE: {self.name} | SUCCESS ====")
