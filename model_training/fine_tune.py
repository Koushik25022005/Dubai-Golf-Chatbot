import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Default model, can be overridden. For demonstration, we use a smaller model.
MODEL_NAME = "gpt2" # Replace with "meta-llama/Meta-Llama-3-8B-Instruct" for actual use if resources allow
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "lora_model")

def generate_dummy_dataset():
    # In a real scenario, this would load multi-shot QA pairs from the knowledge base
    data = {
        "text": [
            "User: What are the green fees at Dubai Golf? Assistant: The green fees at Dubai Golf vary by season and course. Please check the website for current rates.",
            "User: Do you offer golf lessons? Assistant: Yes, the Peter Cowen Academy offers golf lessons for all skill levels.",
            "User: Where is the club located? Assistant: Dubai Golf operates several clubs including Emirates Golf Club and Dubai Creek Golf & Yacht Club."
        ]
    }
    return Dataset.from_dict(data)

def fine_tune():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    
    # Setup PEFT / LoRA
    print("Configuring LoRA for PEFT...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"], # Adjust depending on the model (e.g., q_proj, v_proj for Llama)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Dataset
    dataset = generate_dummy_dataset()
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=10,
        # Remove mixed precision for basic CPU demo, but add fp16=True for GPU
    )
    
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    fine_tune()
