import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import os
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = r'C:\path\to\fine_tuned_model'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Simple Dataset for Curriculum Learning
class SimpleDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        inputs = self.tokenizer(line, return_tensors="pt", max_length=128, padding=True, truncation=True)
        return {'input_ids': inputs['input_ids'].squeeze(0), 'attention_mask': inputs['attention_mask'].squeeze(0)}

# Load simple dataset first
dataset = SimpleDataset(r'C:\path\to\simple_dataset.txt', tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Fine-tune the model with simple data first
def fine_tune_simple():
    model.train()
    for epoch in range(2):  # Start with fewer epochs
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")

fine_tune_simple()

# Now use a more complex dataset for further training
complex_dataset = SimpleDataset(r'C:\path\to\complex_dataset.txt', tokenizer)
complex_dataloader = DataLoader(complex_dataset, batch_size=2, shuffle=True)

# Fine-tune on complex data
def fine_tune_complex():
    model.train()
    for epoch in range(5):
        for batch in complex_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Complex data loss: {loss.item()}")

fine_tune_complex()
