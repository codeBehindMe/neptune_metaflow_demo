import neptune
from metaflow import FlowSpec, step

from src.config.config import Config
from src.data.data import Data
from src.model.iris_model import RFEstimator, SVMEstimator

with open(".neptune_token") as f:
    neptune_token = f.readlines()[0]


project = "atillek/iris"


class PredictFlow(FlowSpec):
    @step
    def start(self):
        self.token = neptune_token

        project = neptune.init_project(project="atillek/iris", mode="read-only")

        runs_table_df = project.fetch_runs_table()
