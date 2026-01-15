from MLP import MLP
from MLP.utils import PipeValues

class Pipeline:
    transformers = {}
    train_values = None
    base_values = None
    steps = None

    def __init__(self, steps):
        self.steps = steps
        self.transformers = {name : transformers for name, transformers in steps}

    def fit(self, X, y):
        self.train_values = PipeValues(X.copy(), y.copy())
        self.base_values = PipeValues(X.copy(), y.copy())
        for i, (name, transformer) in enumerate(self.transformers.items()):
            if i < len(self.transformers.keys()) - 1:
                transformer.fit_transform(self.train_values)
            else:
                transformer.fit(self.train_values.X, self.train_values.y)

    def predict(self, X):
        values = PipeValues(X.copy(), self.base_values.y)

        for i, (name, transformer) in enumerate(self.transformers.items()):
            if i < len(self.transformers.keys()) - 1:
                transformer.transform(values)
            else:
                values.y = transformer.predict(values.X)
        for i, (name, transformer) in reversed(list(enumerate(self.transformers.items()))):
            if i < len(self.transformers.keys()) - 1:
                transformer.inverse_transform(values)
        return values.y

    def copy(self):
        new_pipeline = Pipeline(self.steps)

        new_pipeline.transformers = {name: transformers.copy() for name, transformers in zip(self.transformers.keys(), self.transformers.values())}
        new_pipeline.steps = self.steps.copy()
        new_pipeline.train_values = self.train_values.copy()
        new_pipeline.base_values = self.base_values.copy()
        return new_pipeline



def make_pipeline(*steps):
    return Pipeline([(type(x).__name__, x) for x in steps])
