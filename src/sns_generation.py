from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_sns_post(processed_text):
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    inputs = tokenizer(processed_text, return_tensors='pt')
    
    # Ensure `attention_mask` is provided and set `pad_token_id`
    attention_mask = inputs['attention_mask']
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(inputs['input_ids'], attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=150)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
