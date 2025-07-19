from datasets import load_dataset


def load_dataset_sentimental(dataset_name:str="imdb"):
    try:
        dataset = load_dataset(path=dataset_name)
        return dataset
    except Exception as e:
        raise ValueError(f"Erro ao carregar o dataset {dataset_name}! Erro: {e}\n")