from collections import OrderedDict
import warnings

import flwr as fl
import torch
import numpy as np
import pandas as pd

import random
from torch.utils.data import DataLoader

import sys

from datasets import load_dataset, load_metric, Dataset

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

warnings.filterwarnings("ignore", category=UserWarning)
partition_id = int(sys.argv[1])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "xlm-roberta-base"  # transformer model checkpoint

def clientID(id):
    match id:
        case 1:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu1.csv"
        case 2:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu2.csv"
        case 3:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu3.csv"
        case 4:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu4.csv"
        case 5:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu5.csv"
        case 6:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu6.csv"
        case _:
            print("Value Must Be 1 through 6")
            quit()
            return None
            
            
def testID(id):
    match id:
        case 1:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu1.csv"
        case 2:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu2.csv"
        case 3:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu3.csv"
        case 4:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu4.csv"
        case 5:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu5.csv"
        case 6:
            return "C:\\Users\\karen\\Desktop\\Thesis\\Machine Learning\\urdu6.csv"
        case _:
            print("Value Must Be 1 through 6")
            quit()
            return None


def load_data():
    csvFile = clientID(partition_id)
    
    df = pd.read_csv(csvFile)

    raw_datasets = Dataset.from_pandas(df)

    raw_datasets = raw_datasets.shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["message"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns("message")
    tokenized_datasets = tokenized_datasets.rename_column("spam/ham", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets,
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )
    
    csvFile = testID(partition_id)
    
    df = pd.read_csv(csvFile)

    raw_datasets = Dataset.from_pandas(df)

    raw_datasets = raw_datasets.shuffle(seed=42)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns("message")
    tokenized_datasets = tokenized_datasets.rename_column("spam/ham", "labels")


    testloader = DataLoader(
        tokenized_datasets, batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


def main():
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    trainloader, testloader = load_data()

    # Flower client
    class FLClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(net, trainloader, epochs=1)
            print("Training Finished.")
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("localhost:8080", client=FLClient(),grpc_max_message_length = 1122211497)


if __name__ == "__main__":
    main()