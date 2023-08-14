import os

import neptune
import pandas as pd
from metaflow import FlowSpec, step

from src.data.data import Data
from src.model.iris_model import RFEstimator, SVMEstimator

with open(".neptune_token") as f:
    neptune_token = f.readlines()[0]


project = "atillek/iris"


def get_run_object_at_runtime(run_id: str):
    return neptune.init_run(
        with_id=run_id, mode="read-only", api_token=neptune_token, project=project
    )


class PredictFlow(FlowSpec):
    @step
    def start(self):
        self.token = neptune_token

        neptune_project = neptune.init_project(
            project=project, mode="read-only", api_token=neptune_token
        )
        runs_table_df = neptune_project.fetch_runs_table().to_pandas()
        self.latest_run_id = runs_table_df["sys/id"][0]

        self.next(self.prep_inference_data)

    @step
    def prep_inference_data(self):
        self.x_predict, _ = Data().get_test()

        self.next(self.predict_svm)

    @step
    def predict_svm(self):
        run = get_run_object_at_runtime(self.latest_run_id)
        run["svm/weights"].download("svm")

        svm = SVMEstimator()
        svm.load("svm")
        os.remove("svm")

        print(svm.predict(self.x_predict))

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PredictFlow()
