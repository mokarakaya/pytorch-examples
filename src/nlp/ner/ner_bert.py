import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
from src.util import get_device, set_seed

device = get_device()
set_seed()
LEARNING_RATE = 1e-2
EPOCHS = 5

# Download data set from: https://www.kaggle.com/datasets/rajnathpatel/ner-data

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")


def align_label(texts, labels, labels_to_ids, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        texts, padding="max_length", max_length=512, truncation=True
    )

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(
                    labels_to_ids[labels[word_idx]] if label_all_tokens else -100
                )
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df, labels_to_ids):
        lb = [i.split() for i in df["labels"].values.tolist()]
        txt = df["text"].values.tolist()
        self.texts = [
            tokenizer(
                str(i),
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for i in txt
        ]
        self.labels = [align_label(i, j, labels_to_ids) for i, j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels


def main():
    df = pd.read_csv("~/develop/dataset/ner/ner.csv")
    df = df[0:1000]
    labels = [i.split() for i in df["labels"].values.tolist()]

    # Check how many labels are there in the dataset
    unique_labels = set()

    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]

    # Map each label into its id representation and vice versa
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    labels = [i.split() for i in df["labels"].values.tolist()]

    # Check how many labels are there in the dataset
    unique_labels = set()

    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]
    model = BertModel(unique_labels)

    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    def train_loop(model, df_train, df_val):

        train_dataset = DataSequence(df_train, labels_to_ids)
        val_dataset = DataSequence(df_val, labels_to_ids)

        train_dataloader = DataLoader(
            train_dataset, num_workers=4, batch_size=1, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=1)

        optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

        model = model.to(device)

        best_acc = 0
        best_loss = 1000

        for epoch_num in range(EPOCHS):

            total_acc_train = 0
            total_loss_train = 0

            model.train()

            for train_data, train_label in tqdm(train_dataloader):
                train_label = train_label[0].to(device)
                mask = train_data["attention_mask"][0].to(device)
                input_id = train_data["input_ids"][0].to(device)

                optimizer.zero_grad()
                loss, logits = model(input_id, mask, train_label)

                logits_clean = logits[0][train_label != -100]
                label_clean = train_label[train_label != -100]

                predictions = logits_clean.argmax(dim=1)

                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

                loss.backward()
                optimizer.step()

            model.eval()

            total_acc_val = 0
            total_loss_val = 0

            for val_data, val_label in val_dataloader:
                val_label = val_label[0].to(device)
                mask = val_data["attention_mask"][0].to(device)

                input_id = val_data["input_ids"][0].to(device)

                loss, logits = model(input_id, mask, val_label)

                logits_clean = logits[0][val_label != -100]
                label_clean = val_label[val_label != -100]

                predictions = logits_clean.argmax(dim=1)

                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

            val_accuracy = total_acc_val / len(df_val)
            val_loss = total_loss_val / len(df_val)

            print(
                f"Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}"
            )

    train_loop(model, df_train, df_val)


class BertModel(torch.nn.Module):
    def __init__(self, unique_labels):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(
            "bert-base-cased", num_labels=len(unique_labels)
        )

    def forward(self, input_id, mask, label):

        output = self.bert(
            input_ids=input_id, attention_mask=mask, labels=label, return_dict=False
        )

        return output


if __name__ == "__main__":
    main()
