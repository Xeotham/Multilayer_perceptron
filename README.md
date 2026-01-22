# Multilayer_perceptron
Project to create our very first Neural Network, a Multilayer Perceptron.
# Sources:
**_Multilayer Perceptron_**
- [FreeCodeCamp article on how multilayer perceptron work](https://www.freecodecamp.org/news/build-a-multilayer-perceptron-with-examples-and-python-code/)
- [Playlist of video from Machine Learnia about DeepLearning](https://www.youtube.com/watch?v=XUFLq6dKQok&list=PLO_fdPEVlfKoanjvTJbIbd9V5d9Pzp8Rw)
- [Video from IBM youtube channel that explain Multilayer Perceptron](https://www.youtube.com/watch?v=7YaqzpitBXw)
- [Very clear video of why we need multilayer perceptron](https://www.youtube.com/watch?v=u5GAVdLQyIg)
- [3Blue1Brown video about neural network](https://www.youtube.com/watch?v=aircAruvnKk&t=4s)

**_Feature Selection_**
- [Article about the Variance Threshold](https://www.geeksforgeeks.org/machine-learning/variance-threshold/)
- [Video about the Mutual Information](https://www.youtube.com/watch?v=eJIp_mgVLwE)

# Usage

## Setup
You first need to **setup** the **virtual environment** and **import** the **dataset**.
To do so, **execute** this line on your **terminal** on the **root folder**.

```shell
> source ./setup.sh
```

It will create the **virtual environment** and **import** the **dataset**.

## Split the data
Once done, you now can execute the ``split_data.py`` file.

```shell
> python split_data.py [OPTION]... [Dataset Path]
```
```
Options:
    --test_name=<string> (default: test_set.csv) : Name of the test set file.
    --train_name=<string> (default: train_set.csv) : Name of the train set file.
    --test_size=<float | int | None> (default: None) : Size of the test set.
    --train_size=<float | int | None> (default: None) : Size of the train set.
    --shuffle=<bool> (default: True) : Shuffle the test set.
    --seed=<int> (default: None) : Random seed for shuffling.
```

This scrypt will take the **Dataset** passed as argument and **split** it into a **Train set** and a **Test set**.

## Train the model
Once your **dataset** divided, you can now train the **model**. To do so, execute the ``train_model.py`` scrypt. It'll **prompt** you information to **setup** your **Neural Network**.
```shell
> python train_model.py [Path to the Trainset]
```
Once **done training**, the scrypt will save the model on a `Pickle` file.

## Predict from the model
Since you now have a **trained model** and a **test dataset**, you can use them to make prediction.
To do so, use the ``predict.py`` scrypt.
```shell
> python predict.py [Path to the pkl model] [Path to the test set]
```

It'll use **LogLoss** to check the cost of the test.

# Multilayer Perceptron
## Concept and use
The **Multilayer Perceptron** is one of the first ever **Deep Neural Network**.
It's divided into **Layers** which are also divided in what's called a **Perceptron**.

A **perceptron** is a simple function that take **different parameters**, **Weights** and **Bias** as **input**, do the **Weighted sum** and give one **output**. This output then pass into an **activation function** like a **sigmoid** or **ReLu** activation function.
The **Weight** and **Bias** are values that will **change** on the **learning process**

This **activation function** will **place** the **output** between **0** and **1**.
The **activation function** is only **useful** in the case of a **classification algorithm**. A **regression** type **Neural Network** won't use one.
![Image of a Perceptron](https://media.geeksforgeeks.org/wp-content/uploads/20251209120638608023/bhu.webp)

A **Layer** is a group of **Perceptron**. Each **Perceptron** take the same **inputs** but different **weight** and **bias**, and give their **output** to the next **Layer**.

A full **Neural Network** is composed of an **input layer**, which are the **X**, one or multiple **hidden layers**, which will transform their given **input** to a **new output** and connect to the **next layer** (hidden or not), and an **output layer** which will give the **final prediction**.

![Image of a Neural Network](https://cdn-images-1.medium.com/max/800/0*eaw1POHESc--l5yR.png)

The process that goes from the **input layer** to the **output layer** is called **Forward Propagation**, but to **learn** a **neural network** need to **minimize** it's **cost**.
To do so, we use different **cost function** depending on the context (**Binary LogLoss** for binary prediction or **Multiclass LogLoss** for multiclass prediction for example).
With the **cost function** we can calculate our **cost**, that's good but to learn we need to **minimize our cost**. To do so, we can **derive our cost function** for the **Weight** and for the **bias**, it'll give us **the gradient**.

The **gradient** give us the **way we** need to go and the **learning rate** tell us how much we need to move. With those, we use what's called an **optimizer** like the **Gradient Descent** to modify our **Weights** and **bias** accordingly to minimize the sayed **cost**.
![Image of the Gradient Descent](https://datacorner.fr/wp-content/uploads/2021/03/gradient_descent_1.jpg)


With all that, we have the **Backward Propagation**.
![Image of forward and backward propagation](https://www.researchgate.net/publication/363833676/figure/fig4/AS:11431281086090733@1664108836658/the-Forward-and-Backward-Propagation-in-ANN.jpg)

Now that we understand how a **Multilayer Perceptron** work, what is its use ?

With a **Multilayer Perceptron**, we can solve **Regression Problem** (predict value of a car, predict cost of a House, etc...) and **Classification Problem** (Guess the type of animal, Guess the type of disease, etc...), that are **non-linear**, with simple linear **neurones**.
Since the **Multilayer Perceptron** is one of the first, it may not be suited for more complex task like **image recognition**, where a **Convolutional Neural Network** work better, but it's **simple to build**, **fast** and mostly **efficient**.