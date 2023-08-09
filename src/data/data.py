from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self) -> None:
        db = load_iris()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            db.data, db.target
        )

    def get_train(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.x_train, self.y_train

    def get_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.x_test, self.y_test
