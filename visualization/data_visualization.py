from pandas import DataFrame, read_csv
from pandas.errors import EmptyDataError
from matplotlib.pyplot import subplots, show
from seaborn import pairplot
from numpy import ndarray, vstack

# def scatter_plot(y: ndarray, X_b: ndarray, X_m: ndarray):
#     fig, ax = subplots()
#     print(y[y == 1].shape)
#     print(y[y == 0].shape)
#     print(X_b.shape)
#     print(X_m.shape)
#     ax.scatter(X_b[:, 0], y[y == 1], c='b')
#     ax.scatter(X_m[:, 0], y[y == 0], c='r')
#     show()

def show_pair_plot(df: DataFrame):
    df = df.set_axis(["species"] + list(range(df.shape[1] - 1)), axis='columns')
    pairplot(df, hue="species")
    show()

def main():
    try:
        df: DataFrame = read_csv('../data/data.csv', header=None, index_col=None)
        y = df.iloc[:, 1].copy()
        y[y == 'M'] = 0
        y[y == 'B'] = 1
        X = df.iloc[:, 2:].copy()
        X_b = X[y == 1]
        X_m = X[y == 0]
        # show_pair_plot(df.iloc[:, 1:].copy())
        print (df.iloc[:, 1:].describe())
        return
    except AssertionError as msg:
        print(msg)
        return
    except FileNotFoundError:
        print("FileNotFoundError: provided file not found.")
    except PermissionError:
        print("PermissionError: permission denied on provided file.")
    except EmptyDataError:
        print("EmptyDataError: Provided dataset is empty.")
    except KeyError as err:
        print(f"KeyError: {err} is not in the required file.")
    return


if __name__ == "__main__":
    main()