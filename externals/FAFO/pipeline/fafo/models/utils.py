
def build_prompt(model_name, tokenizer, prompt, conversation):
    if len(conversation) == 0:
        conversation.append({"role": "system", "content": "You are a useful assistant."})
    conversation.append({"role": "user", "content": prompt})

    return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True), conversation
