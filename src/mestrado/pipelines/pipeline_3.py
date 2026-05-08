from src.pipelines.base_pipeline import BasePipeline

class Pipeline3(BasePipeline):
    
    #Pipeline do PKDD

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.name = config["name"]

    def run_steps(self):
        self.logger.info(f"==== START PIPELINE: {self.name} ====")

        self.run_preprocessing()
        self.run_text_polarization()
        self.run_sampling()
        self.run_image_to_text()
        self.run_scoring()

        self.logger.info(f"==== END PIPELINE: {self.name} | SUCCESS ====")

