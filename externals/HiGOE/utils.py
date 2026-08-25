import os
import json
import pickle
import time
import datetime
import random

import torch
import openai
import networkx as nx
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def show_time():
    time_stamp = '\033[1;31;40m[' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ']\033[0m'

    return time_stamp


def get_device(index=0):
    return torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")


def text_wrap(text):
    return '\033[1;31;40m' + str(text) + '\033[0m'


def load_nx(path) -> nx.Graph:
    return nx.read_graphml(path)


def store_nx(nx_obj, path):
    nx.write_graphml_lxml(nx_obj, path)


def write_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def write_to_pkl(data, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)


def read_from_pkl(output_file):
    with open(output_file, 'rb') as file:
        data = pickle.load(file)

    return data


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def print_metrics(metrics):
    for k, v in metrics.items():
        ff = "{} " + k + " ("
        metric = metrics[k]
        for sub_k in metric.keys():
            ff += sub_k + "/"
        ff = ff[:-1] + "): "
        for sub_v in metric.values():
            ff += format(sub_v, ".4f") + "/"
        ff = ff[:-1]
        print(ff.format(show_time()))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LocalLLMManager:
    _instance = None
    _model = None
    _tokenizer = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalLLMManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path):
        if self._model is None or self._model_path != model_path:
            print(f"Loading model from {model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self._model_path = model_path
            print("Model loaded successfully!")
        return self._tokenizer, self._model

llm_manager = LocalLLMManager()

def get_llm_response_via_local(prompt,
                               MODEL_PATH="Llama-2-7b-chat",
                               MAX_LENGTH=2048,
                               TEMPERATURE=1.0,
                               TOP_P=1.0,
                               DO_SAMPLE=True,
                               SEED=42):
    
    if SEED is not None:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
    
    try:
        tokenizer, model = llm_manager.load_model(MODEL_PATH)
        
        system_message = "You are a helpful assistant that provides accurate and concise summaries."
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=3072)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=MAX_LENGTH,  
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1, 
                length_penalty=1.0,      
            )
        
        generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        print(f"Error in local model generation: {e}")
        raise e



def get_llm_response_via_api1(prompt,
                             API_BASE="xxx",
                             API_KEY="xxx",
                             LLM_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1",
                             TAU=1.0,
                             TOP_P=1.0,
                             N=1,
                             SEED=42,
                             MAX_TRIALS=5,
                             TIME_GAP=5):
    '''
    res = get_llm_response_via_api(prompt='hello')  # Default: TAU Sampling (TAU=1.0)
    res = get_llm_response_via_api(prompt='hello', TAU=0)  # Greedy Decoding
    res = get_llm_response_via_api(prompt='hello', TAU=0.5, N=2, SEED=None)  # Return Multiple Responses w/ TAU Sampling
    '''
    openai.api_base = API_BASE
    openai.api_key = API_KEY
    completion = None
    while MAX_TRIALS:
        MAX_TRIALS -= 1
        try:
            completion = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                n=N,
                temperature=TAU,
                top_p=TOP_P,
                seed=SEED,
            )
            break
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception("Reach MAX_TRIALS={}".format(MAX_TRIALS))
    contents = completion.choices
    if len(contents) == 1:
        return contents[0].message["content"]
    else:
        return [c.message["content"] for c in contents]

# self-hosted model
def get_llm_response_via_api(prompt,
                             API_BASE="xxx",
                             API_KEY="xxx",
                             LLM_MODEL="gpt-4",
                             TAU=1.0,
                             TOP_P=1.0,
                             N=1,
                             SEED=42,
                             MAX_TRIALS=5,
                             TIME_GAP=5):
    '''
    res = get_llm_response_via_api(prompt='hello')  # Default: TAU Sampling (TAU=1.0)
    res = get_llm_response_via_api(prompt='hello', TAU=0)  # Greedy Decoding
    res = get_llm_response_via_api(prompt='hello', TAU=0.5, N=2, SEED=None)  # Return Multiple Responses w/ TAU Sampling
    '''
    openai.api_base = API_BASE
    openai.api_key = API_KEY
    completion = None
    while MAX_TRIALS:
        MAX_TRIALS -= 1
        try:
            completion = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                n=N,
                temperature=TAU,
                top_p=TOP_P,
                seed=SEED,
            )
            break
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception("Reach MAX_TRIALS={}".format(MAX_TRIALS))
    
    if isinstance(completion, str):
        return completion

    contents = completion.choices
    if len(contents) == 1:
        return contents[0].message.content
    else:
        return [c.message.content for c in contents]

if __name__ == '__main__':
    pass
