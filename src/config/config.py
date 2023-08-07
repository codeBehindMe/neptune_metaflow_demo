import yaml


class Config:
    def __init__(self) -> None:
        pass

    def load_config(self):
        return yaml.load("config.yaml")
