from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def generate_paragraph(prompt_text):
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    
    output_ids = model.generate(
        input_ids,
        max_length=150,      
        num_return_sequences=1,  
        no_repeat_ngram_size=2,  
        early_stopping=True
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text
user_input = input("Enter some text: ")

generated_paragraph = generate_paragraph(user_input)

print("\nGenerated Paragraph:\n")
print(generated_paragraph)
