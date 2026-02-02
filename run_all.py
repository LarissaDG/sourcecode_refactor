from src.utils.logging import setup_logger
from src.pipelines.pipeline_1 import Pipeline1
from src.pipelines.pipeline_2 import Pipeline2
from src.pipelines.pipeline_3 import Pipeline3

logger = setup_logger("master", "logs")

pipelines = [
    Pipeline1(config1, logger),
    Pipeline2(config2, logger),
    Pipeline3(config3, logger),
]

for pipeline in pipelines:
    try:
        pipeline.run()
    except Exception:
        logger.error(
            f"PIPELINE FAILED: {pipeline.name}",
            exc_info=True
        )
