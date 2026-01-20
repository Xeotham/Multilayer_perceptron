from getopt import getopt, GetoptError
from ML_MHAOUAS import train_test_split
from sys import argv
from pandas import DataFrame, read_csv

class Arguments:
    test_set_name: str = "./data/test_set.csv"
    train_set_name: str = "./data/train_set.csv"
    test_size: float | int | None = None
    train_size: float | int | None = None
    shuffle: bool = True
    shuffle_seed: int = None

input_args = Arguments

def get_flags():
    """
    :return:
    """

    args = argv[1:]

    long_flags: list[str] = ["test_name=", "train_name=", "test_size=", "train_size=", "shuffle=", "seed=", "help"]

    opts, args = getopt(args, "", long_flags)

    def check_size_format(s: str):
        value_type = None

        if s == "None":
            return None
        if s.find(".") != s.rfind("."):
            return False
        for c in s:
            if c in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9") and value_type is None:
                value_type = int
            elif c == ".":
                value_type = float
            elif c.isalpha():
                return False
        return value_type


    for opt, arg in opts:
        if opt in ("--test_name"):
            input_args.test_set_name = arg
        elif opt in ("--train_name"):
            input_args.train_set_name = arg
        elif opt in ("--test_size"):
            arg_type = check_size_format(arg)
            assert arg_type is not False, "Error: test_size must be a int, float or None."
            if arg_type is None:
                input_args.test_size = None
            else:
                input_args.test_size = arg_type(arg)
        elif opt in ("--train_size"):
            arg_type = check_size_format(arg)
            assert arg_type is not False, "Error: test_size must be a int, float or None."
            if arg_type is None:
                input_args.train_size = None
            else:
                input_args.train_size = arg_type(arg)
        elif opt in ("--shuffle"):
            if arg == "True":
                input_args.shuffle = True
            elif arg == "False":
                input_args.shuffle = False
            else:
                raise AssertionError("Error: Shuffle must be True or False.")
        elif opt in ("--seed"):
            assert arg.isdigit(), "Error: seed must be an integer."
            input_args.shuffle_seed = int(arg)
        elif opt in ("--help"):
            print(
"""python split_data.py [OPTION]... [Dataset Path]

    --test_name=<string> (default: test_set.csv) : Name of the test set file.
    --train_name=<string> (default: train_set.csv) : Name of the train set file.
    --test_size=<float | int | None> (default: None) : Size of the test set.
    --train_size=<float | int | None> (default: None) : Size of the train set.
    --shuffle=<bool> (default: True) : Shuffle the test set.
    --seed=<int> (default: None) : Random seed for shuffling.
""")
            return False
    return argv[len(opts) + 1:]

def main():
    try:
        arguments = get_flags()

        if arguments is False:
            return
        elif not arguments:
            raise AssertionError("Error: No dataset provided.")
        assert len(arguments) == 1, "Error: Only one dataset can be provided."

        df = read_csv(arguments[0], header=None, index_col=None)
        test_set, train_set = train_test_split(df, test_size=input_args.test_size, train_size=input_args.train_size, shuffle=input_args.shuffle, random_state=input_args.shuffle_seed)
        test_set.to_csv(input_args.test_set_name, header=0, index=0)
        train_set.to_csv(input_args.train_set_name, header=0, index=0)
    except AssertionError as err:
        print(err)
    except GetoptError:
        print("python3 [flags]\n")


if __name__ == "__main__":
    main()