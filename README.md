# DQN - SNAKE

AI for Snake game trained using Deep Reinforcement Learning (DQN).
The brain of the game was built using Neural Networks and Reinforcement Learning.
The project used the Python Learning Environment in order to display the graphics of the game, and to build the vision of the snake, or what the snake sees.


# State & Neural Network
Our neural network received as input a bunch o states in order to predict the next best action.
We have considered that a state should contain the following items:
* x coordinate of head
* y coordinate of head
* x coordinate of apple
* y coordinate of apple

# Requirements
The project was made using Python 3.7, Python Learning Environment, Numpy and Keras.
The core of the game is in the new_snake.py file, which uses the following command line arguments:
* the learning rate hyperparameter used by the neural network
* gamma parameter used by the agent
* the name of the model
* the activation for the first hidden layer from the network
* the activation for the second hidden layer from the network
* test/train, if you want to test the model, or if you want to train it



## Results
The maximum score achieved by our model was 38.  We have used the following parameters:


|learning rate  |0.004 |
|---------------|------|
|discount factor|0.95  |
|activation#1   |linear|
|activation#2   |linear|
|batch size     |100   |
|memory buffer  |500000|
|epsilon        |0.99  |
|epsilon decay  |0.9999|


The file trained_model.h5.png contains a chart that displays some experimental results made during the test phase.
