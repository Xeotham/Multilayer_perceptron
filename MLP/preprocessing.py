from multiprocessing import Pool
from numpy.random import shuffle as np_shuffle, seed
from numpy import arange, concatenate, mean, ndarray, array, unique, zeros
from pandas import DataFrame
from MLP import MLP

def train_test_split(
    *array_lst,
    test_size: int | float | None = None,
    train_size: int | float | None = None,
    random_state = None,
    shuffle:bool = True
):
    """
    Function to split the data into train and test sets.
    :param array_lst: Arrays of the same length or same shape[0].
    :param test_size: Size of the test set.
                      The value can be ``None``, ``float`` (Value between 0 and 1, proportionality of the dataset) or
                      ``int`` (Literal size of the dataset).
                      If ``None``, take the rest of ``train_size``.
                      If ``train_size`` is None, value is 0.25.
    :param train_size: Size of the train set.
                      The value can be ``None``, ``float`` (Value between 0 and 1, proportionality of the dataset) or
                      ``int`` (Literal size of the dataset).
                      If ``None``, take the rest of ``test_size``.
    :param random_state: Seed for shuffling the dataset.
    :param shuffle: Boolean flag to shuffle the dataset.
    :return: List of all the train and test sets.
    """

    def get_sizes(array_len):
        """
        Function to get the size of test_set and train_set.
        :param array_len: Length of the array, or it shape[0]
        :return: Size of the test and train set.
        """
        check_value = lambda x: x * array_len if isinstance(x, float) else x if isinstance(x, int) else None
        true_test_size, true_train_size = check_value(test_size), check_value(train_size)
        if true_test_size is None and true_train_size is None:
            true_test_size = int(array_len * 0.25)
            true_train_size = int(array_len * 0.75)
        elif true_test_size is None and true_train_size is not None:
            true_test_size = array_len - true_train_size
        elif true_test_size is not None and true_train_size is None:
            true_train_size = array_len - true_test_size
        elif true_test_size > array_len or true_train_size > array_len:
            tmp = array_len * true_test_size / (true_test_size + true_train_size)
            true_train_size = array_len * true_train_size / (true_test_size + true_train_size)
            true_test_size = tmp
        elif true_test_size + true_train_size > array_len:
            tmp = ((true_test_size + true_train_size) - array_len) // 2
            true_test_size = int(true_test_size - tmp)
            true_train_size = int(true_train_size - tmp)
        return int(true_test_size), int(true_train_size)
    def get_array_len():
        """
        Function to check and get the size of the array_lst.
        :return: Final size of the array.
        """
        check_size = lambda xx: xx.shape[0] if isinstance(xx, (ndarray, DataFrame)) else len(xx)
        ref_size = check_size(array_lst[0])
        for x in array_lst[1:]:
            assert isinstance(x, (ndarray, DataFrame, list, tuple))
            size = check_size(x)
            assert size == ref_size, "Error: Provided array size is not equal."
        return ref_size
    def check_param_error():
        """
        Function to check the parameters of the function.
        :return:
        """
        assert len(array_lst) > 0, "Error: No array provided."
        for x in array_lst:
            assert isinstance(x, (ndarray, DataFrame, list, tuple)), "Error: Provided array is not an array."
        assert isinstance(test_size, (int, float)) or test_size is None, "Error: test_size must be of type int, float or None."
        assert isinstance(train_size, (int, float)) or train_size is None, "Error: train_size must be of type int, float or None."
        assert isinstance(shuffle, bool), "Error: Shuffle must be of type bool."

    train_test_lst = []

    try:
        check_param_error()

        test_size, train_size = get_sizes(get_array_len())
        for arr in array_lst:
            if isinstance(arr, (tuple, list)):
                new_arr = array(arr)
            else:
                new_arr = arr.copy()
            if shuffle:
                if isinstance(new_arr, DataFrame):
                    new_arr = new_arr.sample(frac=1, random_state=random_state).reset_index(drop=True)
                else:
                    seed(random_state)
                    np_shuffle(new_arr)
            if isinstance(new_arr, DataFrame):
                train_test_lst.append(new_arr.iloc[:test_size, :])
                train_test_lst.append(new_arr.iloc[test_size:test_size + train_size, :].reset_index(drop=True))
            else:
                train_test_lst.append(new_arr[0:test_size])
                train_test_lst.append(new_arr[test_size:test_size + train_size])
        return train_test_lst
    except AssertionError as err:
        print(err)
    except NameError as err:
        print(err)
    except KeyError as err:
        print(err)

def _verif(iter_number: int, fold_size: int, X_shuffled, y_shuffled, ml_class) -> float:
        """
        Evaluates the model on a single fold.

        Args:
            iter_number (int): Index of the current fold.

        Returns:
            float: Accuracy of the model on the validation fold.
        """
        val_start = iter_number * fold_size
        val_end = (iter_number + 1) * fold_size
        X_val = X_shuffled[val_start:val_end]
        y_val = y_shuffled[val_start:val_end]
        X_train = concatenate([X_shuffled[:val_start], X_shuffled[val_end:]])
        y_train = concatenate([y_shuffled[:val_start], y_shuffled[val_end:]])

        class_copy = ml_class.copy()

        # Train the model and generate predictions
        class_copy.fit(X_train, y_train)
        y_pred = class_copy.predict(X_val)
        accuracy = mean(y_pred == y_val)
        return accuracy

def cross_validation(
    X: ndarray,
    y: ndarray,
    ml_class: MLP,
    k: int = 5,
    multi_process: bool = False,
) -> float:
    """
    Performs k-fold cross-validation for a multinomial logistic regression model.

    Args:
        X (ndarray): Input feature matrix of shape (n_samples, n_features).
        y (ndarray): Target labels of shape (n_samples,).
        k (int, optional): Number of folds for cross-validation. Defaults to 5.
        multi_process (bool, optional): If True, a calling process use multiprocess. Defaults to False.

    Returns:
        float: Average accuracy of the model across all folds.
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = arange(n_samples)
    np_shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    accuracies = []

    if not multi_process:
        # Use multiprocessing for cross-validation
        with Pool(processes=k) as pool:
            print("Enter in pool")
            results = [pool.apply_async(_verif, (i, fold_size, X_shuffled, y_shuffled, ml_class)) for i in range(k)]
            accuracies = [i.get() for i in results]
    else:
        for i in range(k):
            accuracies.append(_verif(i))

    return mean(accuracies)
