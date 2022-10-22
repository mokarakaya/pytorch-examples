import torch
import torch.nn.functional as F

from src.data_util import get_movielens
from src.movielens.collaborative_filtering.collab_constants import MODEL_PATH
from src.movielens.collaborative_filtering.collab_model import CollabFNet
from src.util import get_device, set_seed

device = get_device()


df_train, df_val = get_movielens()
num_users = df_train["user_id"].max() + 1
num_items = df_train["item_id"].max() + 1
print(num_users, num_items)

model = CollabFNet(num_users, num_items, emb_size=100).to(device)  # .cuda()


def get_test_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(df_val["user_id"].values).to(device)
    items = torch.LongTensor(df_val["item_id"].values).to(device)
    ratings = torch.FloatTensor(df_val["rating"].values).to(device)
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    # print("test loss %.3f " % loss.item())
    return loss.item()


def train_epocs(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    set_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for i in range(epochs):
        model.train()
        users = torch.LongTensor(df_train["user_id"].values).to(device)
        items = torch.LongTensor(df_train["item_id"].values).to(device)
        ratings = torch.FloatTensor(df_train["rating"].values).to(device)
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_test_loss = get_test_loss(model, unsqueeze)
        print(i, loss.item(), current_test_loss)
    return model


if __name__ == "__main__":
    model = train_epocs(model, epochs=500, lr=0.001, wd=1e-6, unsqueeze=True)
    torch.save(model.state_dict(), MODEL_PATH)
    # 999 0.8575757145881653 0.8811629414558411
