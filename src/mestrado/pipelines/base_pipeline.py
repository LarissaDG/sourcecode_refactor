class BasePipeline:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.name = config["name"]

    def run(self):
        self.logger.info(f"==== START PIPELINE: {self.name} ====")
        self.run_steps()
        self.logger.info(f"==== END PIPELINE: {self.name} | SUCCESS ====")

    def run_steps(self):
        raise NotImplementedError
