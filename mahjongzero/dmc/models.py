"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn

winds = ['east', 'south', 'west', 'north']

class WindLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

# Model dict is only used in evaluation but not training
model_dict = {w: WindLstmModel for w in winds}

class Model:
    """
    The wrapper for the four models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models = {w: WindLstmModel().to(torch.device(device)) for w in winds}

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        for w in winds:
            self.models[w].share_memory()

    def eval(self):
        for w in winds:
            self.models[w].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
