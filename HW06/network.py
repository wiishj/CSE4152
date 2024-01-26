import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ClassificationNetworkColors(torch.nn.Module):
    def __init__(self):

        super().__init__()
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classes = [[-1., 0., 0.],  # left
                        [-1., 0.5, 0.], # left and accelerate
                        [-1., 0., 0.8], # left and brake
                        [1., 0., 0.],   # right
                        [1., 0.5, 0.],  # right and accelerate
                        [1., 0., 0.8],  # right and brake
                        [0., 0., 0.],   # no input
                        [0., 0.5, 0.],  # accelerate
                        [0., 0., 0.8]]  # brake

        """
        D : Network Implementation

        Implementation of the network layers. 
        The image size of the input observations is 96x96 pixels.

        Using torch.nn.Sequential(), implement each convolution layers and Linear layers
        """

        # convolution layers 
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7056, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)
        self.epsilon=1e-5

        # Linear layers (output size : 9)





    def forward(self, observation):
        #B,W,H,C->B,C,W,H
        x=observation.permute(0, 3, 1, 2)
        """
        D : Network Implementation

        The forward pass of the network. 
        Returns the prediction for the given input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
         

    def actions_to_classes(self, actions):
        """
        C : Conversion from action to classes

        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector 
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1
        """
        classes=[]
        for i in actions:
            for j in range(len(self.classes)):
                tmp=self.classes[j]
                tmp=torch.Tensor(tmp)
                result=i-tmp
                result=torch.abs(result)
                result=torch.sum(result)
                if(result<=self.epsilon) : classes+=torch.tensor([[j]])
        return classes

    def scores_to_action(self, scores):
        """
        C : Selection of action from scores

        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        return self.classes[torch.argmax(scores, keepdim=True)]

