# DQN - SNAKE

AI for Snake game trained using Deep Reinforcement Learning (DQN).
The brain of the game was built using Neural Networks and Reinforcement Learning.
The project used the Python Learning Environment in order to display the graphics of the game, and to build the vision of the snake, or what the snake sees.


# State & Neural Network
Our neural network received as input a bunch o states in order to predict the next best action.
We have considered that a state should contain the following items:
* 1 or 0 if the apple is in front of the snake
* 1 or 0 if there is an obstacle in front of the snake
* 1 or 0 if the apple is at the left of the snake
* 1 or 0 if there is an obstacle at the left of  the snake
* 1 or 0 if the apple is at the right of the snake
* 1 or 0 if there is an obstacle at the right of the snake

We have managed the build this kind of *vision* for the snake using the head coordinates, apple coordinates and body part coordinates. 

Also, the direction of the snake is really important, because based on that we will decide which key to be pressed in order to perform a certain action. The possible actions in our case are: go straight forward, turn left, turn right. And this is also the output of our network. Our network will tell us if we should take a move(move left or right) or do nothing.

We have said that direction plays a crucial because, if we are heading for example to the left side, this means that if I keep going forward, i will be going to the left direction, if I want to go Left, in fact I will go down, and if I want to go up, I will go Right. This is happening because we are choosing the action based on the relative direction of the snake. Which means, in the case we are going Left, then if we go to the Left of the snake we will, in fact, go DOWN. 

This is important because we have considered that WASD mean exactly the directions: Forward, Left, Down, Right. That means that if I want to go left, normally  I would press A. But that is not always True. If I am heading Left, and I want to Left again, key A won`t work, because we want to go Left *of the snake*. So, LEFT of the snake is Down, and that is letter S.

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
