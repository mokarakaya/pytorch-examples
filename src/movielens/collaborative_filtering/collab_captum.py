import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients

from src.data_util import get_movielens
from src.movielens.collaborative_filtering.collab_constants import MODEL_PATH
from src.movielens.collaborative_filtering.collab_model import CollabFNet
from src.util import get_device


def get_model(df_train, device):
    num_users = df_train["user_id"].max() + 1
    num_items = df_train["item_id"].max() + 1
    print(num_users, num_items)

    model = CollabFNet(num_users, num_items, emb_size=100).to(device)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def main():
    device = "cpu"
    df_train, df_val = get_movielens()
    model = get_model(df_train, device)

    users = torch.IntTensor(df_val["user_id"][:2].values).to(device)
    items = torch.IntTensor(df_val["item_id"][:2].values).to(device)
    # ratings = torch.FloatTensor(df_val["rating"].values).to(device)
    ratings = df_val["rating"].astype(int).tolist()
    lig = LayerIntegratedGradients(model, [model.user_emb, model.item_emb])
    # attr, delta = lig.attribute((users, items), target=ratings, return_convergence_delta=True)
    attr, delta = lig.attribute((users, items), return_convergence_delta=True)
    attributions = attr[0]
    attributions = attributions.sum().squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    # importances = np.mean(attr, axis=0)
    print(attributions)


if __name__ == "__main__":
    main()
