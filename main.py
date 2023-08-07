import fire


class ModelExecutionContainer:
    def train(self):
        print("training started")

    def predict(self):
        print("batch prediction started")

    def server(self):
        print("Server started")


if __name__ == "__main__":
    fire.Fire(ModelExecutionContainer)
