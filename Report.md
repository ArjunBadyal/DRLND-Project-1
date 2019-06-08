[//]: # (Image References)

[image1]: https://github.com/arjunlikesgeometry/DRLND-Project-1/blob/master/P1.png
[image2]: https://github.com/arjunlikesgeometry/DRLND-Project-1/blob/master/DQN.png

### Introduction
In this project the DQN algorithm was used to solve the environment. In particular DQN makes use of a fixed Q target and an experience replay buffer. The action value function is approxmimated using a neural net.

### Algorithm and Network Architecture
![DQN][image2]
The algorithm above was taken from this <cite><a href="https://arxiv.org/abs/1509.02971"><i>paper</i></a></cite>; outlining the use of experience replay and a fixed Q-target to perform the gradient descent update step on the weights in the neural net used to approximate the action value function. The overall code was modified for this environment from that given in the udacity lesson on DQN.

The neural net used just three linear layers and the relu activation function. 

```python
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

The hyperparameters were as follows:
```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
```

### Results
The results below show that the environment was solved in 224 episodes i.e. this was the point where the average score over the last 100 episodes was greater than or equal to 13. The weights used at this point have been saved in the checkpoint.pth file and may be loaded to see the performance of the trained model. 


Episode 100	Average Score: 3.62

Episode 200	Average Score: 7.81

Episode 300	Average Score: 11.99

Episode 324	Average Score: 13.06

Environment solved in 224 episodes!	Average Score: 13.06

![Trained Agent][image1]


### Conclusion and Future Work
Although the environment was solved relatively quickly the algorithm could have been improved by optomizing the hyperparameters further. Other variations of DQN such as double DQN could also be used to improve maximum reward as well as the time the algorithm takes to train to. Futher considerations could aslo have been taken to address issues of stability such as convergence and divergence after a large number of episodes when considering the maximum posible reward obtainable by the agent. 

Future work could include solving some of the issues outlined above as well as applying DQN to the same environment, but with raw pixel data as input.
