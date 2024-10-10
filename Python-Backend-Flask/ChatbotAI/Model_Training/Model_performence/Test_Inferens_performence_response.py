from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_path = r'C:\Program Files (x86)\LearnReflect Project\LearnReflect Project\Python-Backend-Flask\ChatbotAI\gpt2-medium-model'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = 50256


input_text = "Hello"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

output = model.generate(input_ids, attention_mask=attention_mask, min_length=2,max_length=5)
print(tokenizer.decode(output[0], skip_special_tokens=True))


response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
