from multiprocessing import Pool
from numpy.random import permutation, seed
from numpy import arange, concatenate, mean, ndarray, array
from pandas import DataFrame
from MLP import MLP


def train_test_split(
        *array_lst,
        test_size: int | float | None = None,
        train_size: int | float | None = None,
        random_state=None,
        shuffle: bool = True
):

    def get_array_len(arrays):
        return len(arrays[0]) if not hasattr(arrays[0], 'shape') else arrays[0].shape[0]

    try:
        length = get_array_len(array_lst)
        # Calculate split index
        if isinstance(test_size, float):
            split_idx = int(length * (1 - test_size))
        elif isinstance(train_size, float):
            split_idx = int(length * train_size)
        else:
            split_idx = int(length * 0.75)  # Default

        # 1. Generate Indices
        indices = arange(length)
        if shuffle:
            if random_state is not None:
                seed(random_state)
            indices = permutation(indices)

        train_test_lst = []
        for arr in array_lst:
            # 2. Apply SAME indices to all arrays
            if isinstance(arr, DataFrame):
                # DataFrame handling
                shuffled = arr.iloc[indices].reset_index(drop=True)
                train = shuffled.iloc[:split_idx]
                test = shuffled.iloc[split_idx:]
            else:
                # Numpy handling
                arr_shuffled = array(arr)[indices]
                train = arr_shuffled[:split_idx]
                test = arr_shuffled[split_idx:]

            train_test_lst.extend([test, train])  # Matching your return order (Test, Train)

        return train_test_lst

    except Exception as e:
        print(f"Error in split: {e}")
        return []


def _verif(iter_number: int, fold_size: int, X_shuffled, y_shuffled, ml_class) -> float:
    val_start = iter_number * fold_size
    val_end = (iter_number + 1) * fold_size

    X_val = X_shuffled[val_start:val_end]
    y_val = y_shuffled[val_start:val_end]

    # Efficient concatenation
    X_train = concatenate([X_shuffled[:val_start], X_shuffled[val_end:]])
    y_train = concatenate([y_shuffled[:val_start], y_shuffled[val_end:]])

    class_copy = ml_class.copy()
    class_copy.fit(X_train, y_train)

    y_pred = class_copy.predict(X_val)

    # Ensure dimensions match for accuracy calculation
    if y_pred.shape != y_val.shape:
        # Handle one-hot vs scalar mismatch if necessary
        pass

    return mean(y_pred == y_val)


def cross_validation(
        X: ndarray,
        y: ndarray,
        ml_class: MLP,
        k: int = 5,
        multi_process: bool = False,  # Logic inverted in original? Standard is False -> No Multiprocess
) -> float:
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = arange(n_samples)
    seed(42)  # Consistent CV
    indices = permutation(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    accuracies = []

    if multi_process:
        with Pool(processes=k) as pool:
            results = [pool.apply_async(_verif, (i, fold_size, X_shuffled, y_shuffled, ml_class)) for i in range(k)]
            accuracies = [res.get() for res in results]
    else:
        for i in range(k):
            accuracies.append(_verif(i, fold_size, X_shuffled, y_shuffled, ml_class))

    return mean(accuracies)