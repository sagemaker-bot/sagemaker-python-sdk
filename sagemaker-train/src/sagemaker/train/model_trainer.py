import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, role=None, source_code=None, compute=None, **kwargs):
        self.role = role
        self.source_code = source_code
        self.compute = compute

    def train(self, **kwargs):
        logger.info("Starting training")
        return {"status": "completed"}
