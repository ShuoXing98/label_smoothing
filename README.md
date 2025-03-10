# label_smoothing

if script_args.add_chatbot_arena_data:
    def tokenize_add_data(example):
        kwargs = {"truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
        prompt, response_a, response_b = example['conversation_a'][0]['content'], example['conversation_a'][1]['content'], example['conversation_b'][1]['content']
        messages = [ {"content": prompt, "role": "user" }, { "content": '\n\n##Response A: {} \n\n##Response B: {}'.format(response_a, response_b), "role": "assistant" } ]
        prompt_plus_response = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode_plus(prompt_plus_response, **kwargs)
        example["input_ids"] = tokens["input_ids"][0]
        example["attention_mask"] = tokens["attention_mask"][0]
        if example['winner'] == 'model_a':
            example['labels'] = torch.tensor([0.9, 0.05, 0.05]).float()
        elif example['winner'] == 'model_b':
            example['labels'] = torch.tensor([0.05, 0.9, 0.05]).float()
        else:
            example['labels'] = torch.tensor([0.05, 0.05, 0.9]).float()
        return example
