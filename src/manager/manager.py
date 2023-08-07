from typing import Any
from src.data.data import Data


class TrainManager:
    def __init__(self) -> None:
        pass

    def train(self):
        x_train, y_train = Data().get_train()


class PredictionManager:
    def __init__(self) -> None:
        pass

    def predict(self):
        pass


class ServerManager:
    def __init__(self) -> None:
        pass
