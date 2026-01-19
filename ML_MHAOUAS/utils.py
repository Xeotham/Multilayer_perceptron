from numpy import exp, ndarray, empty_like, std, mean, array, max, sum
from pickle import load, dump
from pathlib import Path


def sigmoid(
    z: ndarray
) -> ndarray:
    """
    Applies a sigmoid function h to each value of the matrix z,
    where h(x) = 1 / (1 + e^(x))

    Args:
        z: A matrix

    Returns:
        ndarray: A new matrix where each element v has been mapped to h(v)
    """
    # print(f"Shape of Z: {z.shape}, Z: {type(z[0, 0])}")
    out = empty_like(z)
    pos_mask = (z >= 0)
    out[pos_mask] = 1 / (1 + exp(-z[pos_mask]))
    neg_mask = ~pos_mask
    exp_z = exp(z[neg_mask])
    out[neg_mask] = exp_z / (1 + exp_z)

    return out

sigmoid.derivative = lambda z: z * (1 - z)

def relu(
    z: ndarray
) -> ndarray:
    out = z.copy()
    pos_mask = (z <= 0)
    out[pos_mask] = 0
    return out

relu.derivative = lambda z: (z > 0).astype(float)

def softmax(z: ndarray) -> ndarray:
    # Subtract max for numerical stability (prevents exp overflow)
    shift_z = z - max(z, axis=0, keepdims=True)
    exps = exp(shift_z)
    return exps / sum(exps, axis=0, keepdims=True)

softmax.derivative = lambda z: z * (1 - z)

class PipeValues:
    X = None
    y = None
    def __init__(self, X: ndarray, y: ndarray):
        self.X = X
        self.y = y

    def copy(self):
        return PipeValues(self.X, self.y)

def save_in_file(filename: str, data):
    with Path(filename) as output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        file = output_file.open(mode="wb")
        dump(data, file)

def load_from_file(filename: str):
    with open(filename, "rb") as f:
        data = load(f)
        return data