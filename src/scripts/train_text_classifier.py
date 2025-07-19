import torch
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy_with_logits 
from utils.load_dataset_sentimental import load_dataset_sentimental
from models.text_classifier import TEXT_CLASSIFIER
if __name__=="__main__":

    dataset = load_dataset_sentimental()
    small_train_dataset = dataset["train"].shuffle(seed=42).select([i for i in list(range(500))])
    small_test_dataset = dataset["test"].shuffle(seed=42).select([i for i in list(range(100))])
    # Instanciar um modelo
    bert_class_model = TEXT_CLASSIFIER(bert_model_name="distilbert-base-uncased", num_labels=2)
    # Treinar o modelo
    trained_model = bert_class_model.train_process(train_dataset=small_train_dataset, epochs=3)
    # Avaliar o modelo
    bert_class_model.evaluate_process(eval_dataset=small_test_dataset)

    bert_class_model.compute_metrics(eval_dataset=small_test_dataset)
    
    # Salvar o modelo
    torch.save(trained_model, "./data/modelo.pt")