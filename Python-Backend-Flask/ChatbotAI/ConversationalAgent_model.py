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

# Dataset for Predefined Scenarios
class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer):
        self.conversations = conversations
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        prompt, response = self.conversations[idx]
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, padding=True, truncation=True)
        outputs = self.tokenizer(response, return_tensors="pt", max_length=128, padding=True, truncation=True)
        return {'input_ids': inputs['input_ids'].squeeze(0), 'attention_mask': inputs['attention_mask'].squeeze(0), 'labels': outputs['input_ids'].squeeze(0)}

# Predefined conversation scenarios
conversations = [
    ("What is discipline?", "Discipline is the practice of training people to obey rules or a code of behavior."),
    ("How do I improve self-discipline?", "Start by setting clear goals and building small, consistent habits."),
    # Add more scenarios here
]

dataset = ConversationDataset(conversations, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Fine-tune the model with conversations
def train_conversational():
    model.train()
    for epoch in range(3):
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")

train_conversational()
