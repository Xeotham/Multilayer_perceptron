from ML_MHAOUAS import MLP, Layers, save_in_file
from ML_MHAOUAS.Preprocessing import LabelBinarizer, StandardScaler
from ML_MHAOUAS.FeatureSelection import VarianceThreshold, MutualInformation
from sys import argv
from pandas import DataFrame, read_csv
from datetime import datetime

from ML_MHAOUAS.Pipeline import make_pipeline


class Arguments:
    # Datasets
    train_set: str                              = "./data/train_set.csv"

    # FeatureSelection
    VarianceThreshold: VarianceThreshold = VarianceThreshold()
    MutualInformation: MutualInformation = MutualInformation()

    # Global MLP Information
    learning_rate: float                        = 0.001
    epochs: int                                 = 1000
    batch_size: int                             = 32
    patience: int                               = 500

    # Layers Configurations
    layers: list                                = []

    # Plots
    show_plots: bool                            = False

class LayersParameters:
    node_nb: int    = 10,
    activation: str = "relu",
    optimizer: str  = "adam",
    name: str       = "Hidden Layer"

input_args = Arguments

def input_VarianceThreshold():
    input_res = input("Whould you like to use Variance Threshold [y/N]: ")
    if input_res.lower() == "y":
        input_args.VarianceThreshold = VarianceThreshold()
    elif input_res.lower() == "n" or input_res == "":
        input_args.VarianceThreshold = None
    else:
        print("Please enter y or n")
        input_VarianceThreshold()

def input_MutualInformation():
    def input_nb_of_features():
        nb_of_features = input("How many features would you like to use?: ")
        if nb_of_features.isdigit():
            return int(nb_of_features)
        else:
            print("Please enter number of features")
            input_MutualInformation()

    input_res = input("Whould you like to use Mutual information [y/N]: ")
    if input_res.lower() == "y":
        input_args.MutualInformation = MutualInformation(input_nb_of_features())
    elif input_res.lower() == "n" or input_res == "":
        input_args.MutualInformation = None
    else:
        print("Please enter y or n")
        input_MutualInformation()

def input_learning_rate():
    learning_rate = input("Which learning rate would you like to use? [default: 0.001]: ")
    try:
        if learning_rate == "":
            input_args.learning_rate = 0.001
        else:
            input_args.learning_rate = float(learning_rate)
    except ValueError:
        print(f"Enter a valid learning rate (float).")
        input_learning_rate()

def input_epochs():
    epochs = input("How many epochs would you like to use? [default: 1000]: ")
    if epochs == "":
        input_args.epochs = 1000
    elif epochs.isdigit():
        input_args.epochs = int(epochs)
    else:
        print(f"Enter a valid number of epochs (int).")
        input_epochs()

def input_batch_size():
    batch_size = input("How batch would you like to use? [default: 32]: ")
    if batch_size == "":
        input_args.batch_size = 32
    elif batch_size.isdigit():
        input_args.batch_size = int(batch_size)
    else:
        print(f"Enter a valid number of batch_size (int).")
        input_batch_size()

def input_patience():
    patience = input("How much patience would you like to use? [default: 500]: ")
    if patience == "":
        input_args.patience = 500
    elif patience.isdigit():
        input_args.patience = int(patience)
    else:
        print(f"Enter a valid patience (int).")
        input_patience()

def input_nb_neurones():
    node_nb = input("How many neurones would you like to use? on this layer [default: 10]: ")
    if node_nb == "":
        return 10
    elif node_nb.isdigit():
        return int(node_nb)
    else:
        print("Enter a valid number of neurons (int).")
        return input_nb_neurones()

def input_activation():
    possible_activations = ["relu", "sigmoid", "softmax"]
    act = input(f"Which activation would you like to use ({','.join(possible_activations)})? [default: relu]: ")
    if act == "":
        return "relu"
    elif act.lower() in possible_activations:
        return act.lower()
    else:
        print(f"Enter a valid activation type (relu, sigmoid, softmax).")
        return input_activation()

def input_optimizer():
    possible_activations = ["adam", "gd", "momentum"]
    act = input(f"Which optimizer would you like to use ({','.join(possible_activations)})? [default: adam]: ")
    if act == "":
        return "adam"
    elif act.lower() in possible_activations:
        return act.lower()
    else:
        print(f"Enter a valid activation type (adam, gd, momentum).")
        return input_activation()

def input_layer_name():
    act = input("Which name would you like to give your layer?: ")
    return act

def print_layer(idx, layer):

    box_shape = (21, 5)

    print(f"+{f' {idx} '.center(box_shape[0], '-')}+")
    print(f"| {('Name: ' + layer.name[:box_shape[0] - 9] + '.').ljust(box_shape[0] - 2, ' ')} |")
    print(f"| {('Node NB: ' + str(layer.node_nb)).ljust(box_shape[0] - 2, ' ')} |")
    print(f"| {('Activation: ' + layer.activation).ljust(box_shape[0] - 2, ' ')} |")
    print(f"| {('Optimizer: ' + layer.optimizer).ljust(box_shape[0] - 2, ' ')} |")
    print(f"| {' ' * (box_shape[0] - 2)} |")
    print(f"+{'-' * box_shape[0]}+")

def layers_confirm():
    def confirm_layers():
        mod_layer = input("Would you like to modify a layer? [idx/N]")
        print(len(input_args.layers))
        if mod_layer.lower() == "n" or mod_layer == "":
            return
        elif mod_layer.isdigit() and int(mod_layer) < len(input_args.layers):
            input_args.layers[int(mod_layer)] = create_layer()
            return layers_confirm()
        else:
            print("Please enter a valid input")
            return confirm_layers()

    for i, layer in enumerate(input_args.layers):
        print_layer(i, layer)
        print()

    confirm_layers()

def create_layer():
    new_layer_params = LayersParameters()
    new_layer_params.node_nb = input_nb_neurones()
    new_layer_params.activation = input_activation()
    new_layer_params.optimizer = input_optimizer()
    new_layer_params.name = input_layer_name()
    return new_layer_params

def input_layers():
    new_layer = True

    while new_layer:
        input_args.layers.append(create_layer())
        in_new_layer = input("Would you like to add another layer? [Y/n]: ")
        if in_new_layer.lower() == "n":
            new_layer = False
    print("Setup of the Output Layer:")
    input_args.layers.append(create_layer())
    layers_confirm()

def input_plots():
    ret = input("Would you like to see the cost evolution plots? [Y/n]: ")

    if ret.lower() == "y" or ret == "":
        input_args.show_plots = True
    elif ret.lower() == "n":
        input_args.show_plots = False
    else:
        print("Please enter y or n")
        input_plots()

def configure_model():
    # FeatureSelection
    input_VarianceThreshold()
    input_MutualInformation()
    # configure models
    input_learning_rate()
    input_epochs()
    input_patience()
    input_batch_size()
    # configure Layers
    input_layers()
    # Show Plots
    input_plots()

def configure_layers():
    layers = []
    for layer in input_args.layers:
        layers.append(Layers(layer.node_nb, activation=layer.activation, optimizer=layer.optimizer, name=layer.name))
    return tuple(layers)

def main():
    try:
        assert len(argv) == 2, "Error: expected a DataSet as a second argument."
        input_args.train_set = argv[1]
        configure_model()

        df = read_csv(input_args.train_set).values

        y = df[:, 1]
        X = df[:, 2:]

        layers = configure_layers()

        mlp = MLP(
            hidden_layers = layers,
            learning_rate = input_args.learning_rate,
            epochs = input_args.epochs,
            batch_size = input_args.batch_size,
            patience = input_args.patience
        )

        args = [StandardScaler(), LabelBinarizer()]
        if input_args.VarianceThreshold is not None:
            args.append(input_args.VarianceThreshold)
        if input_args.MutualInformation is not None:
            args.append(input_args.MutualInformation)
        args.append(mlp)
        print(args)
        pipeline = make_pipeline(*args)

        print("Starting Training...")
        pipeline.fit(X.astype(float), y)

        if input_args.show_plots:
            mlp.show_plots()
        print("Training Complete!")

        save_path = f"./saved_models/MLP_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        print(f"Saving model as {save_path}.")
        save_in_file(save_path, pipeline)
    except AssertionError as err:
        print(err)
    except RuntimeError as err:
        print(err)
    except KeyboardInterrupt:
        print("Bye Bye :)!")

if __name__ == "__main__":
    main()
