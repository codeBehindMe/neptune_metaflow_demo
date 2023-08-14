import pandas as pd
import neptune
from metaflow import FlowSpec, step

from src.config.config import Config
from src.data.data import Data
from src.model.iris_model import RFEstimator, SVMEstimator

with open(".neptune_token") as f:
    neptune_token = f.readlines()[0]


project = "atillek/iris"


def _get_latest_run(runs_table : pd.DataFrame):
  latest_run_id =runs_table['sys/id'][0]
  

class PredictFlow(FlowSpec):
    @step
    def start(self):
        self.token = neptune_token

        neptune_project = neptune.init_project(
            project=project, mode="read-only", api_token=neptune_token
        )

        runs_table_df = neptune_project.fetch_runs_table().to_pandas()

        print(runs_table_df["sys/id"][0])
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PredictFlow()
