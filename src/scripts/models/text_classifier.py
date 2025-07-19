import torch
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy_with_logits
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score

class TEXT_CLASSIFIER:
    def __init__(self, bert_model_name:str="distilbert-base-uncased",num_labels:int=2):
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Carregar o tokenizer e modelo
        self.bert_model_name=bert_model_name
        self.num_labels=num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.bert_model_name, num_labels=self.num_labels).to(self.device)

        
    def train_process(self, train_dataset, epochs):
        opt = Adam(self.model.parameters(), lr=2e-5)
        self.model.train()
        for epoch_index in range(epochs):
            total_loss = 0
            for sample in train_dataset:
                inputs = self.tokenizer(sample['text'], padding=True, truncation=True, return_tensors="pt").to(self.device)
                labels = torch.tensor([sample['label']]).to(self.device)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {epoch_index+1}/{epochs}, Loss: {total_loss/len(train_dataset)}")

        return self.model

    def evaluate_process(self, eval_dataset):
        self.model.eval()
        val_loss = []
        for sample in eval_dataset:
            inputs = self.tokenizer(sample['text'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            labels = torch.tensor([sample['label']]).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
            val_loss.append(loss.item())
        val_loss_value = sum(val_loss)/len(val_loss)
        print(f"Validation loss: {val_loss_value}")

        return val_loss_value

                
    def compute_metrics(self, eval_dataset):

        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        for sample in eval_dataset:
            inputs = self.tokenizer(sample['text'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            labels = torch.tensor([sample['label']]).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits) if self.num_labels == 1 else torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1) if self.num_labels > 1 else (probs > 0.5).long()
            all_labels.append(labels.item())
            all_preds.append(pred.item())
            if self.num_labels == 2:
                all_probs.append(probs[:, 1].item())
            else:
                all_probs.append(probs.squeeze().item())

        accuracy = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = None

        metrics = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "f1_score": f1,
            "recall": recall,
            "auc": auc
        }
        print("Metrics:", metrics)
        return metrics