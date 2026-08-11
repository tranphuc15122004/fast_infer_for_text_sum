import atexit

import torch
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM

from rag_baseline.rag_utils import (retrieve_query_fn,
                                    retrieve_relevant_sentences)


class RagLlama:
    def __init__(
        self, 
        llama_model_name: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct', 
        percentage: float = 0.5
    ) -> None:
        self.llama = LlamaForCausalLM.from_pretrained(
            llama_model_name, 
            torch_dtype=torch.float16, 
            attn_implementation="flash_attention_2", 
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        self.percentage = percentage

        # stats related stuffs
        self.reset_stats()
        atexit.register(self.print_stats)

    def reset_stats(self):
        self.num_queries = 0
        self.ratio = 0

    def print_stats(self):
        if self.num_queries == 0:
            print("Currently no processed queries. ")
            avg_ratio = None
        else:
            avg_ratio = self.ratio / self.num_queries
            print(f"Processed {self.num_queries} queries with avg {avg_ratio * 100:.2f}% keep ratio.")
        return self.num_queries, avg_ratio

    def update_stats(self, ratio):
        self.num_queries += 1
        self.ratio += ratio

    @torch.inference_mode
    def generate(
        self, 
        context: str, 
        input: str, 
        prompt_format: str, 
        dataset_name: str, 
        max_gen: int, 
        apply_chat_template: bool
    ): 
        # query used for retrieval
        query = retrieve_query_fn(dataset_name=dataset_name)(input)

        # calculate original length of the prompt
        ori_prompt = prompt_format.format(input=input, context=context)
        if apply_chat_template:
            messages = [{'role': 'user', 'content': ori_prompt}]
            ori_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        inputs = self.tokenizer([ori_prompt], return_tensors='pt')
        ori_len = inputs['input_ids'].shape[1]

        # calculate retrieval length
        ret_len = int(ori_len * self.percentage)

        # rag part
        ret_context = retrieve_relevant_sentences([context], [query], token_budgets=[ret_len])[0]

        # assemble new prompt
        rag_prompt = prompt_format.format(input=input, context=ret_context)
        if apply_chat_template:
            messages = [{'role': 'user', 'content': rag_prompt}]
            rag_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        inputs = self.tokenizer([rag_prompt], return_tensors='pt')
        input_ids = inputs['input_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        final_len = input_ids.shape[1]
        
        gen_config = GenerationConfig(
            do_sample=False, 
            eos_token_id=128009, 
            pad_token_id=128009
        )

        # update stats
        self.update_stats(final_len / ori_len)

        outputs = self.llama.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  
            max_new_tokens=max_gen, 
            return_dict_in_generate=True, 
            generation_config=gen_config
        )

        generated_tokens = outputs.sequences[:, final_len:]

        return self.tokenizer.decode(generated_tokens.tolist()[0], skip_special_tokens=True)
