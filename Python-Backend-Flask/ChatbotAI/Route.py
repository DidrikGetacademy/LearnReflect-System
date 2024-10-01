import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from flask import Blueprint, request, jsonify
import logging

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

chatbot = Blueprint('chatbot', __name__)

# Load fine-tuned model and tokenizer
model_path = r"C:\Users\Didrik\OneDrive\Skrivebord\LearnReflect Project\Python-Backend-Flask\ChatbotAI\fine_tuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Define Custom Dataset Class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Save feedback for future fine-tuning
def save_feedback(response_text, feedback_score):
    with open('feedback_data.txt', 'a') as f:
        f.write(f"{response_text}\t{feedback_score}\n")
    logging.info('Feedback saved: Response: %s, Score: %d', response_text, feedback_score)

# Update model immediately with positive feedback
def update_model_immediately(response_text):
    encodings = tokenizer(
        [response_text],  
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    
    dataset = CustomDataset(encodings)
    
    training_args = TrainingArguments(
        output_dir=r'C:\Users\Didrik\OneDrive\Skrivebord\LearnReflect Project\Python-Backend-Flask\ChatbotAI\fine_tuned_model',
        num_train_epochs=1, 
        per_device_train_batch_size=1,
        learning_rate=2e-5,
        save_steps=100,
        logging_dir='./logs',
        logging_steps=10,
        weight_decay=0.01
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")
    logging.info('Model updated with new feedback.')

@chatbot.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data.get('message', '')

    if not input_text:
        logging.error('Empty message received.')
        return jsonify({"response": "Error: No input text provided."}), 400

    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_return_sequences=1,
        temperature=0.5,
        top_k=30,
        top_p=0.8,
        no_repeat_ngram_size=2,
        do_sample=True
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info('Generated response: %s', response_text)
    
    return jsonify({"response": response_text})

@chatbot.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    response_text = data.get('response', '')
    feedback_score = int(data.get('score', 0))

    if not response_text or feedback_score not in [-1, 1]:
        logging.error('Invalid feedback data: Response: %s, Score: %d', response_text, feedback_score)
        return jsonify({"status": "error", "message": "Invalid feedback data"}), 400

    save_feedback(response_text, feedback_score)
    
    if feedback_score > 0:
        update_model_immediately(response_text)
    
    return jsonify({"status": "success"})
