from MLP import MLP

class Pipeline:
    transformers = {}

    def __init__(self, steps):
        self.transformers = {name : transformers for name, transformers in steps}

    def fit(self, X, y):
        values = [X.copy(), y.copy()]
        for i, (name, transformer) in enumerate(self.transformers.items()):
            if i < len(self.transformers.keys()) - 1:
                transformer._pipe_fit_transform(values)
            else:
                transformer.fit(values[0], values[1])