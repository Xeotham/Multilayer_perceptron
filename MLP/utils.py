from numpy import exp, ndarray, empty_like, std, mean


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

def softmax(z: ndarray) -> ndarray:
    return exp(z) / sum(exp(z))

def standardize(
    x: ndarray,
):
    return (x - mean(x)) / std(x)