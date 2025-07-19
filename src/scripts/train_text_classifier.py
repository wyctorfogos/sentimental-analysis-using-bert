import torch
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy_with_logits
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.load_dataset_sentimental import load_dataset_sentimental

def train_process(train_dataset, model, tokenizer, device, epochs):
    opt = Adam(model.parameters(), lr=2e-5)
    model.train()
    for epoch_index in range(epochs):
        total_loss = 0
        for sample in train_dataset:
            inputs = tokenizer(sample['text'], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = torch.tensor([sample['label']]).to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch_index+1}/{epochs}, Loss: {total_loss/len(train_dataset)}")

    return model

def evaluate_process(eval_dataset, model, tokenizer, device):
    model.eval()
    val_loss = []
    for sample in eval_dataset:
        inputs = tokenizer(sample['text'], padding=True, truncation=True, return_tensors="pt").to(device)
        labels = torch.tensor([sample['label']]).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
        val_loss.append(loss.item())
    val_loss_value = sum(val_loss)/len(val_loss)
    print(f"Validation loss: {val_loss_value}")
    return val_loss_value

if __name__=="__main__":
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Carregar o tokenizer e modelo
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)
    dataset = load_dataset_sentimental()
    small_train_dataset = dataset["train"].shuffle(seed=42).select([i for i in list(range(500))])
    small_test_dataset = dataset["test"].shuffle(seed=42).select([i for i in list(range(10))])
    
    # Treinar o modelo
    trained_model = train_process(train_dataset=small_train_dataset, model=model, tokenizer=tokenizer, device=device, epochs=3)
    # Avaliar o modelo
    evaluate_process(eval_dataset=small_test_dataset, model=trained_model, tokenizer=tokenizer, device=device)
    
    # Salvar o modelo
    torch.save(trained_model, "./data/modelo.pt")