import os

import neptune
from metaflow import FlowSpec, step

from src.config.config import Config
from src.data.data import Data
from src.model.iris_model import RFEstimator, SVMEstimator

with open(".neptune_token") as f:
    neptune_token = f.readlines()[0]


project = "atillek/iris"


class TrainFlow(FlowSpec):
    @step
    def start(self):
        self.token = neptune_token
        run = neptune.init_run(project=project, api_token=neptune_token)

        self.run_id = run["sys/id"].fetch()

        run.stop()
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        run = neptune.init_run(
            project=project, with_id=self.run_id, api_token=neptune_token
        )

        self.train_x, self.train_y = Data().get_train()
        self.test_x, self.test_y = Data().get_test()

        run["data_prep/train_x/instance_count"] = self.train_x.shape[0]
        run["data_prep/train_y/instance_count"] = self.train_y.shape[0]
        run["data_prep/test_x/instance_count"] = self.test_x.shape[0]
        run["data_prep/test_y/instance_count"] = self.test_y.shape[0]

        run.stop()
        self.next(self.train_svm, self.train_rf)

    @step
    def train_svm(self):
        run = neptune.init_run(
            project=project, with_id=self.run_id, api_token=neptune_token
        )

        self.svm = SVMEstimator()
        self.svm.train(self.train_x, self.train_y)

        with open("svm", "wb") as f:
            self.svm.save(f)
        run["svm/weights"].upload("svm")
        os.remove("svm")

        run.stop()
        self.next(self.join)

    @step
    def train_rf(self):
        run = neptune.init_run(
            project=project, with_id=self.run_id, api_token=neptune_token
        )
        self.rf = RFEstimator()
        self.rf.train(self.train_x, self.train_y)

        with open("rf", "wb") as f:
            self.rf.save(f)
        run["rf/weights"].upload("rf")
        os.remove("rf")

        run.stop()

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print("end")


config = Config().load_config()


if __name__ == "__main__":
    TrainFlow()
