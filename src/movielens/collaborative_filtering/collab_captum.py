import torch

from src.data_util import get_movielens
from src.movielens.collaborative_filtering.collab_constants import MODEL_PATH
from src.movielens.collaborative_filtering.collab_model import CollabFNet
from src.util import get_device


def get_model():
    device = get_device()

    df_train, df_val = get_movielens()
    num_users = df_train["user_id"].max() + 1
    num_items = df_train["item_id"].max() + 1
    print(num_users, num_items)

    model = CollabFNet(num_users, num_items, emb_size=100).to(device)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def main():
    model = get_model()


if __name__ == "__main__":
    main()
