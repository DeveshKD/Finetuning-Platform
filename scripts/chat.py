# scripts/chat.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

def chat_with_lora(base_model_id: str, adapter_path: str):
    print(f"Loading Base Model: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    print(f"Loading LoRA Adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("\nModel ready! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        messages = [{"role": "user", "content": user_input}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,    # <--- Force it to stop if it hits EOS
            repetition_penalty=1.15,                # <--- Punish it for repeating <LM>
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        # Decode and strip the prompt
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nModel: {response}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m scripts.chat <base_model> <adapter_path>")
        sys.exit(1)
    chat_with_lora(sys.argv[1], sys.argv[2])