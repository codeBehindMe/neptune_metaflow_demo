import yaml


class Config:
    def __init__(self) -> None:
        pass

    def load_config(self):
        with open("config.yaml") as f:
            return yaml.load(f, yaml.FullLoader)
