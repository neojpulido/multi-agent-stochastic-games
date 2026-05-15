#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import copy


# You need to instantiate the constant for the size of the state space below. This will be used as the size of the input tensor for your Q-network

# In[3]:


statespace_size=7


# The function "prepare_torch" needs to be called once and only once at the start of your program to initialise PyTorch and generate the two Q-networks. It returns the target model (for testing).

# In[7]:


def prepare_torch():
  global statespace_size
  global model, model2
  global optimizer
  global loss_fn
  l1 = statespace_size
  l2 = 24
  l3 = 24
  l4 = 4
  model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4))
  model2 = copy.deepcopy(model)
  model2.load_state_dict(model.state_dict())
  loss_fn = torch.nn.MSELoss()
  learning_rate = 1e-3
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  return model2


# The function "update_target" copies the state of the prediction network to the target network. You need to use this in regular intervals.

# In[10]:


def update_target():
  global model, model2
  model2.load_state_dict(model.state_dict())


# The function "get_qvals" returns a numpy list of qvals for the state given by the argument _based on the prediction network_.

# In[13]:


def get_qvals(state):
  state1 = torch.from_numpy(state).float()
  qvals_torch = model(state1)
  qvals = qvals_torch.data.numpy()
  return qvals


# The function "get_maxQ" returns the maximum q-value for the state given by the argument _based on the target network_.

# In[16]:


def get_maxQ(s):
  return torch.max(model2(torch.from_numpy(s).float())).float()


# The function "train_one_step_new" performs a single training step. It returns the current loss (only needed for debugging purposes). Its parameters are three parallel lists: a minibatch of states, a minibatch of actions, a minibatch of the corresponding TD targets and the discount factor.

# In[19]:


def train_one_step(states, actions, targets, gamma):
  # pass to this function: state1_batch, action_batch, TD_batch
  global model, model2
  state1_batch = torch.cat([torch.from_numpy(s).float() for s in states])
  action_batch = torch.Tensor(actions)
  Q1 = model(state1_batch)
  X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
  Y = torch.tensor(targets)
  loss = loss_fn(X, Y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss.item()


# In[ ]:




