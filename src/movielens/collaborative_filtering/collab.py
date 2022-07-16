import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class CollabFNet(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, n_hidden=10):
        super(CollabFNet, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.lin1 = nn.Linear(emb_size * 2, n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.drop1 = nn.Dropout(0.1)

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        x = F.relu(torch.cat([U, V], dim=1))
        x = self.drop1(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

PATH = '/Users/mokarakaya/develop/dataset/movielens/ml-100k/u.data'
data = pd.read_csv(PATH, names=["user_id", "item_id", "rating", "timestamp"], header=None, delimiter='\t')
np.random.seed(3)
msk = np.random.rand(len(data)) < 0.8
df_train = data[msk].copy()
df_val = data[~msk].copy()

num_users = df_train["user_id"].max() + 1
num_items = df_train["item_id"].max() + 1
print(num_users, num_items)

model = CollabFNet(num_users, num_items, emb_size=100) #.cuda()


def get_test_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(df_val["user_id"].values)  # .cuda()
    items = torch.LongTensor(df_val["item_id"].values)  # .cuda()
    ratings = torch.FloatTensor(df_val["rating"].values)  # .cuda()
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    # print("test loss %.3f " % loss.item())
    return loss.item()

def train_epocs(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for i in range(epochs):
        model.train()
        users = torch.LongTensor(df_train["user_id"].values) # .cuda()
        items = torch.LongTensor(df_train["item_id"].values) #.cuda()
        ratings = torch.FloatTensor(df_train["rating"].values) #.cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_test_loss = get_test_loss(model, unsqueeze)
        print(i, loss.item(), current_test_loss)


train_epocs(model, epochs=1000, lr=0.001, wd=1e-6, unsqueeze=True)
# 500 0.8916721940040588 0.9022499918937683