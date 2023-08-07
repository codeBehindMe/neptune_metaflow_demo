import fire
from src.manager.manager import TrainManager
from src.config.config import Config


config = Config().load_config()


class ModelExecutionContainer:
    def train(self):
        print("training started")
        TrainManager()

    def predict(self):
        print("batch prediction started")

    def server(self):
        print("Server started")


if __name__ == "__main__":
    fire.Fire(ModelExecutionContainer)
