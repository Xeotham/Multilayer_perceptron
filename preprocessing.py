from numpy.random import shuffle as np_shuffle
from numpy import arange, concatenate, mean, ndarray

def cross_validation(
    X: ndarray,
    y: ndarray,
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
    threads = []

    def verif(iter_number: int) -> float:
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

        # Train the model and generate predictions
        # weights = logreg_train(y_train, X_train, multi_process=True)  | Change to adapt to Multilayer Perceptron
        # y_pred = generate_predictions(X_val, weights)                 |
        # accuracy = mean(y_pred == y_val)                              |
        # return accuracy

    # @threaded
    # def thread_verif(iter_number: int, queue: Queue) -> None:
    #     """
    #     Threaded wrapper for evaluating the model on a single fold.
    #
    #     Args:
    #         iter_number (int): Index of the current fold.
    #         queue (Queue): Queue to store the accuracy result.
    #     """
    #     queue.put(verif(iter_number))

    # if not multi_process:                             |
    #     # Use multiprocessing for cross-validation    |
    #     with Manager() as manager:                    |
    #         q = manager.Queue()                       |
    #         for i in range(k):                        |
    #             threads.append(thread_verif(i, q))    | Use Pool instead of threaded
    #         for thread in threads:                    |
    #             thread.join()                         |
    #             accuracies.append(q.get())            |
    # else:                                             |
        # Perform cross-validation sequentially         |
    for i in range(k):
        accuracies.append(verif(i))

    return mean(accuracies)
