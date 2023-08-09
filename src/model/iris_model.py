from abc import ABC

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


class Estimator(ABC):
    def __init__(self):
        pass

    def train(self, x_train, y_train):
        raise NotImplementedError()

    def evaluate(self, x_test, y_test):
        raise NotImplementedError()

    def save(self, buffer):
        raise NotImplementedError()


class RFEstimator(Estimator):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=10)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        y_hat = self.model.predict(x_test)
        return accuracy_score(y_test, y_hat)

    def save(self, buffer):
        dump(self.model, buffer)


class SVMEstimator(Estimator):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        y_hat = self.model.predict(x_test)
        return accuracy_score(y_test, y_hat)

    def save(self, buffer):
        dump(self.model, buffer)


class IrisModel:
    def __init__(self) -> None:
        self.estimators = []

    def init_architecture(self):
        self.estimators.append(RFEstimator())
        self.estimators.append(SVMEstimator())

    def train(self):
        pass
