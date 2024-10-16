from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Load the pre-trained model and tokenizer
model_name = "gpt2-medium"  # You can specify other variants like 'gpt2', 'gpt2-large', etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Step 2: Specify the directory to save the model
save_directory = r"C:\Users\didri\Desktop\kopi av learnreflect\LearnReflect-System\Python-Backend-Flask\ChatbotAI\Model"  # Change this to your desired path

# Step 3: Save the model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
