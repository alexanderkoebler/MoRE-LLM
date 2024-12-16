"""
Copyright 2024 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pickle

class TabModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabModel, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 10)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 50)
        self.activation = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(50, 10)
        self.activation = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(10, 5)
        self.activation = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(5, output_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        x = self.softmax(x)
        return x

class MoE_multirules(nn.Module):
    def __init__(self, gating_model, ml_model, rule_expert, transform=None, device="cuda:0"):
        super(MoE_multirules, self).__init__()
        self.device = device
        self.ml_model = ml_model
        self.rule_expert = rule_expert
        self.gating_model = gating_model
        self.transform = transform

    def forward(self, x):
        gate = self.gating_model(x)
        if self.transform:
            x_ = self.transform(x)
        else:
            x_ = x
        y_hat_ruleset = self.rule_expert(x_)
        mask_support = torch.sum(y_hat_ruleset,-1)
        gate = self.gating_model(x)
        y_ml = self.ml_model(x)

        for i, x_e in enumerate(x_):
            if i == 0: 
                if mask_support[i]:
                    y = (torch.multiply(gate[i,0:1], y_ml[i].unsqueeze(0)) + torch.multiply(gate[i,1:2], y_hat_ruleset[i].unsqueeze(0)))
                else:
                    y = y_ml[i].unsqueeze(0)
            else:
                if mask_support[i]:
                    y = torch.cat((y, torch.multiply(gate[i,0:1], y_ml[i].unsqueeze(0) + torch.multiply(gate[i,1:2], y_hat_ruleset[i].unsqueeze(0)))),dim=0)
                else:
                    y = torch.cat((y, y_ml[i].unsqueeze(0)),dim=0)
        return y, gate, mask_support

    def predict_hard(self, x, th=0.5):
        gate = self.gating_model(x)
        if self.transform:
            x_ = self.transform(x)
        else:
            x_ = x
        y_hat_ruleset = self.rule_expert(x_)
        mask_support = torch.sum(y_hat_ruleset,-1)
        gate = self.gating_model(x)
        y_ml = self.ml_model(x)

        for i, x_e in enumerate(x_):
            if i == 0: 
                if mask_support[i]:
                    y = (torch.multiply(gate[i,0:1] > th, y_ml[i].unsqueeze(0)) + torch.multiply(gate[i,1:2] > th, y_hat_ruleset[i].unsqueeze(0)))
                else:
                    y = y_ml[i].unsqueeze(0)
            else:
                if mask_support[i]:
                    y = torch.cat((y, torch.multiply(gate[i,0:1] > th, y_ml[i].unsqueeze(0) + torch.multiply(gate[i,1:2] > th, y_hat_ruleset[i].unsqueeze(0)))),dim=0)
                else:
                    y = torch.cat((y, y_ml[i].unsqueeze(0)),dim=0)
        return y, gate, mask_support
    
    def save(self, path):
        if os.path.isfile(os.path.join(path, "moe.pth")):
            os.remove(os.path.join(path, "moe.pth"))
            os.remove(os.path.join(path, "moe.pickle"))
        torch.save(self.state_dict(), os.path.join(path, "moe.pth"))

        rule_dict = {'rules': self.rule_expert.rules, 'descriptions': self.rule_expert.descriptions,
        'rule_labels': self.rule_expert.rule_labels, 'samples': self.rule_expert.samples, 'pruning_history': self.rule_expert.pruning_history}

        with open(os.path.join(path, "moe.pickle"), 'wb') as f:
            pickle.dump(rule_dict, f)
    
    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "moe.pth")))

        with open(os.path.join(path, "moe.pickle"), 'rb') as f:
            rule_dict = pickle.load(f)

        self.rule_expert.rules = rule_dict['rules']        
        self.rule_expert.descriptions = rule_dict['descriptions']        
        self.rule_expert.rule_labels = rule_dict['rule_labels']        
        self.rule_expert.samples = rule_dict['samples']        
        self.rule_expert.pruning_history = rule_dict['pruning_history']        

class Gate_multirules(nn.Module):
    def __init__(self, input_dim, output_dim=2, predef_model=None, hidden_dim=8, hard=False, linear='nn', hard_th=0.5, temp=1.0):
        super(Gate_multirules, self).__init__()
        self.temp = temp
        self.hard = hard
        self.hard_th = hard_th
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.linear = linear
        if predef_model != None:
            self.net = predef_model
        elif linear == True:
            self.net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                    nn.Softmax(dim=-1))
        else:
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 2),
                                    nn.Softmax(dim=-1))

    def forward(self, x):
        if self.hard:
            return (self.net(x) > self.hard_th).float()
        else:
            return self.net(x)

class MLExpert(nn.Module):
    def __init__(self, input_dim, predef_model=None, hidden_dim=8, linear=False):
        super(MLExpert, self).__init__()
        self.input_dim = input_dim
        if predef_model != None:
            self.net = predef_model
        elif linear == True:
            self.net = nn.Sequential(nn.Linear(input_dim, 2),
                                    nn.Softmax(dim=-1))
        else:
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 2),
                                    nn.Softmax(dim=-1))

    def forward(self, x):
        return self.net(x)