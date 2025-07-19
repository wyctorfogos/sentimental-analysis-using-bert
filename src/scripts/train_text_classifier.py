import torch
from transformers import AutoModel, AutoTokenizer
import datasets
from datasets import load_dataset


def load_dataset_sentimental(dataset_name:str="imdb"):
    try:
        dataset = load_dataset(path=dataset_name)
        return dataset
    except Exception as e:
        raise ValueError(f"Erro ao carregar o dataset {dataset_name}! Erro: {e}\n")

def train_process():
    try:
        train_dataset = datasets.Dataset()       
    except Exception as e:
        raise ValueError("Erro ao treinar os dados!\n")
    
if __name__=="__main__":
    dataset=load_dataset_sentimental()
    small_train_dataset = dataset["train"].shuffle(seed=42).select([i for i in list(range(3000))])
    small_test_dataset = dataset["test"].shuffle(seed=42).select([i for i in list(range(300))])
