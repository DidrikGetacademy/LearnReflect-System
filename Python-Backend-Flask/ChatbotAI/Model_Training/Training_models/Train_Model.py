import os 
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

model_path = r"C:\Program Files (x86)\LearnReflect Project\LearnReflect Project\Python-Backend-Flask\ChatbotAI\gpt2-medium-model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("daily_dialog")
dataset = dataset['train'].train_test_split(test_size=0.5)  
print(dataset)


def concatenate_dialogues(examples):
    return {"text": [" ".join(dialogue) for dialogue in examples['dialog']]}


dialogue_dataset = dataset.map(concatenate_dialogues, batched=True, remove_columns=['dialog'])


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=56)


tokenized_datasets = dialogue_dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
print(tokenized_datasets['train'][0])

log_dir = r"C:\Program Files (x86)\LearnReflect Project\LearnReflect Project\Python-Backend-Flask\ChatbotAI\Model_Training\Training_logs"
os.makedirs(log_dir, exist_ok=True)  
log_file_path = os.path.join(log_dir, "training_log.txt")

training_args = TrainingArguments(
    output_dir=model_path,
    eval_strategy='epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.02,
    logging_dir=log_dir,
    logging_steps=5,
    logging_first_step=True,
)

train_loss_list = []
eval_loss_list = []
epochs_list = []


def Write_TrainingData(train_loss_list, eval_loss_list, epochs_list): 
    data_dir = r'C:\Program Files (x86)\LearnReflect Project\LearnReflect Project\Python-Backend-Flask\ChatbotAI\Model_Training\Training_data'
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'Diagram_eval_trainloss.txt'), 'a') as file:
        train_loss_str = ", ".join(map(str, train_loss_list))
        eval_loss_str = ", ".join(map(str, eval_loss_list))
        Epoch_str = ", ".join(map(str, epochs_list))
        lines = [f"{train_loss_str}\n", f"{eval_loss_str}\n", f"{Epoch_str}\n"]
        file.writelines(lines)


def adding_loss(loss, eval_loss_value, epoch):
    train_loss_list.append(loss)
    eval_loss_list.append(eval_loss_value)
    epochs_list.append(epoch)


def save_checkpoint(trainer, checkpoint_num, train_loss, eval_loss):
    
    checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-{checkpoint_num}")  
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Logging training information to the specified file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"checkpoint: {checkpoint_num}, Train Loss: {train_loss}, Eval Loss: {eval_loss}, epoch: {epoch}\n")
        print("Logpoint has been logged!")


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
class CustomCallback(TrainerCallback): 

    def __init__(self):
        self.train_loss_list = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
            if 'loss' in logs:
                train_loss_list.append(logs['loss'])
                print(f"Logging step: {state.global_step}, Loss: {logs['loss']}")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    callbacks=[CustomCallback()]

)

for epoch in range(training_args.num_train_epochs):
    print(f"Epoch {epoch + 1} Training...")
    metrics = trainer.train()  

    train_loss = train_loss_list[-1] if train_loss_list else None
    eval_loss = trainer.evaluate()['eval_loss']
    adding_loss(train_loss, eval_loss, epoch + 1)  
    Write_TrainingData(train_loss_list, eval_loss_list, epochs_list)
    save_checkpoint(trainer, epoch + 1, train_loss, eval_loss)  
    print(f"Epoch {epoch + 1}: Checkpoint saved")

final_model_path = os.path.join(training_args.output_dir, f"final_model_epoch_{training_args.num_train_epochs}")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Final model and tokenizer saved to {final_model_path}")
