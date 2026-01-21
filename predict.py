from getopt import getopt, GetoptError
from ML_MHAOUAS import load_from_file
from sys import argv
from pandas import DataFrame, read_csv

def main():
    try:
        assert len(argv) == 3, "Error: A model and a Dataset need to be provided."

        model_path = argv[1]
        dataset_path = argv[2]

        df = read_csv(dataset_path).values

        y = df[:, 1]
        X = df[:, 2:]

        model = load_from_file(model_path)

        mlp = list(model.transformers.values())[-1]
        label_binarizer = list(model.transformers.values())[1]
        y_labeled = label_binarizer.transform(y)
        print(f"Cost: {mlp.val_cost_evolution[-1]}")

        pred = model.predict(X)
        print(y)
        print(pred)

        print(f"Cost: {mlp.log_loss(y_labeled.astype(float).T)}")

    except AssertionError as err:
        print(err)
    except GetoptError:
        print("python3 [flags]\n")


if __name__ == "__main__":
    main()