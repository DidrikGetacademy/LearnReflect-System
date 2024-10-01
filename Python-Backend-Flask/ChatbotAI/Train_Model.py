import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from torch.optim import AdamW

# Check if GPU (CUDA) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = r'C:\Users\Didrik\OneDrive\Skrivebord\LearnReflect Project\Python-Backend-Flask\AI-Chatbot-Model\fine_tuned_model'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Dataset class for handling text data
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=512):
        # Open the file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        encoded = self.tokenizer.encode_plus(
            line,
            truncation=True,
            max_length=self.block_size,
            padding='max_length',  # Ensure padding
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(dim=0)
        attention_mask = encoded['attention_mask'].squeeze(dim=0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


# Custom collate function to pad sequences
def collate_fn(batch):
    input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


# Prepare dataset and dataloader
dataset = TextDataset(r'C:\Users\Didrik\OneDrive\Skrivebord\LearnReflect Project\Python-Backend-Flask\AI-Chatbot-Model\Training\Dataset.txt', tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Setup optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(dataloader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training function
def train(model, dataloader, optimizer, scheduler, epochs=3):
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item()}")

# Train the model
train(model, dataloader, optimizer, scheduler)

# Save the trained model and tokenizer
save_directory = r'C:\Users\Didrik\OneDrive\Skrivebord\LearnReflect Project\Python-Backend-Flask\AI-Chatbot-Model\fine_tuned_model'
os.makedirs(save_directory, exist_ok=True)

try:
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")
except Exception as e:
    print(f"Error saving model or tokenizer: {e}")

# Text generation function
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model
test_prompt = "What is discipline?"
print(f"Generated response: {generate_text(test_prompt)}")
