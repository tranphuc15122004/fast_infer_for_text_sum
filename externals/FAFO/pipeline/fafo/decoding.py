import torch
import warnings
from typing import List, Optional, Union
from transformers.generation.utils import (
    LogitsProcessorList, 
    StoppingCriteriaList, 
    GreedySearchOutput
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria
)
import os

FUNC_MAP = {}
CONFIG_MAP = {}


def get_device():
    if "LOCAL_RANK" not in CONFIG_MAP:
        return 0
    return CONFIG_MAP["LOCAL_RANK"]


def greedy_search_proxy(self, *args, **kwargs):
    USE_FAFO = kwargs['config']['fafo']
    if USE_FAFO:
        return jacobi_greedy_search_multilevel(self, chat=False, *args, **kwargs)
    else:
        return FUNC_MAP["greedy_search"](self, *args, **kwargs)


def update_token_map(token_map, lst_token, past_tokens, new_results, prefix_tokens, LEVEL, WINDOW_SIZE, GUESS_SET_SIZE):
    if GUESS_SET_SIZE != -1: #limited guess set size for each key, LRU policy
        PREFIX_LEN = 3
        prefixes = [(lst_token)]
        prefix = [lst_token]
        if len(prefix_tokens[0]) > 0:
            for j in range(1,PREFIX_LEN):
                prefix = [prefix_tokens[-j][0]] + prefix
                prefixes.append(tuple(prefix))
        for pre in prefixes:
            if pre not in token_map:
                token_map[pre] = []
        
        tup = tuple(past_tokens[ll][0] for ll in range(1, LEVEL - 1)) + (new_results[0],)
        for pre in prefixes:
            if tup in token_map[pre]:
                token_map[pre].remove(tup)
                token_map[pre].append(tup)
            elif len(token_map[pre]) < GUESS_SET_SIZE:
                token_map[pre].append(tup) 
            else:
                assert len(token_map[pre]) == GUESS_SET_SIZE
                token_map[pre] = token_map[pre][1:] + [tup]

        for i in range(1, WINDOW_SIZE):
            prefixes = [(past_tokens[0][i - 1])]
            prefix = [past_tokens[0][i - 1]]
            if len(prefix_tokens[0]) > 0:
                for j in range(1,PREFIX_LEN):
                    prefix = [prefix_tokens[-j][i]] + prefix
                    prefixes.append(tuple(prefix))
            for pre in prefixes:
                if pre not in token_map:
                    token_map[pre] = []

            tup = tuple(past_tokens[ll][i] for ll in range(1, LEVEL - 1)) + (new_results[i],)

            for pre in prefixes:
                if tup in token_map[pre]:
                    token_map[pre].remove(tup)
                    token_map[pre].append(tup)
                elif len(token_map[pre]) < GUESS_SET_SIZE:
                    token_map[pre].append(tup) 
                else:
                    assert len(token_map[pre]) == GUESS_SET_SIZE
                    token_map[pre] = token_map[pre][1:] + [tup]

    else: #unlimited guess set size for each key 
        #first add 
        if lst_token not in token_map:
            token_map[lst_token] = set()
        #build tuple
        tup = tuple(past_tokens[ll][0] for ll in range(1, LEVEL - 1)) + (new_results[0],)
        #add tuple
        token_map[lst_token].add(tup) 

        for i in range(1, WINDOW_SIZE):
            if past_tokens[0][i - 1] not in token_map:
                token_map[past_tokens[0][i - 1]] = set()
            tup = tuple(past_tokens[ll][i] for ll in range(1, LEVEL - 1)) + (new_results[i],)
            token_map[past_tokens[0][i - 1]].add(tup) 

def append_new_generated_pool(tokens, token_map, LEVEL, GUESS_SET_SIZE):
    if len(tokens) != LEVEL:
        return 
    lst_token = tokens[0]
    tup = tuple(tokens[1:])

    if GUESS_SET_SIZE != -1: #limited guess set size for each key, lru policy  
        if lst_token not in token_map:
            token_map[lst_token] = []
        if tup in token_map[lst_token]:
            token_map[lst_token].remove(tup)
            token_map[lst_token].append(tup)
        elif len(token_map[lst_token]) < GUESS_SET_SIZE:
            token_map[lst_token].append(tup) 
        else:
            assert len(token_map[lst_token]) == GUESS_SET_SIZE
            token_map[lst_token] = token_map[lst_token][1:] + [tup]
    else: #unlimited guess set size for each key 
        #first add 
        if lst_token not in token_map:
            token_map[lst_token] = set()
        token_map[lst_token].add(tup) 


def fill_pool_with_prompt(prompts, token_map, LEVEL, GUESS_SET_SIZE):
    for start_idx in range(len(prompts) - LEVEL + 1):
        lst_token = prompts[start_idx]
        tup = tuple(prompts[start_idx+1:start_idx+LEVEL])
        
        if len(tup) != LEVEL - 1:
            return 
        
        if GUESS_SET_SIZE != -1: #limited guess set size for each key, lru policy  
            if lst_token not in token_map:
                token_map[lst_token] = []
            if tup in token_map[lst_token]:
                token_map[lst_token].remove(tup)
                token_map[lst_token].append(tup)
            elif len(token_map[lst_token]) < GUESS_SET_SIZE:
                token_map[lst_token].append(tup) 
            else:
                assert len(token_map[lst_token]) == GUESS_SET_SIZE
                token_map[lst_token] = token_map[lst_token][1:] + [tup]
        else: #unlimited guess set size for each key 
            #first add 
            if lst_token not in token_map:
                token_map[lst_token] = set()
            token_map[lst_token].add(tup) 


def jacobi_greedy_search_multilevel(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    
    chat: bool = False, 
    stop_token: Optional[str]= None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>


    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> outputs = model.greedy_search(
    ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    max_length = CONFIG_MAP['config'].get('max_len', None)
    n_new_tokens = CONFIG_MAP['config'].get('n_new_tokens', None)
    init_len = input_ids.shape[1]
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    ############### configurations 
    WINDOW_SIZE = CONFIG_MAP.get("WINDOW_SIZE", 60)
    GUESS_SET_SIZE = CONFIG_MAP.get("GUESS_SET_SIZE", 60)
    ALWAYS_FWD_ONE = CONFIG_MAP.get("ALWAYS_FWD_ONE", 1)
    LEVEL = CONFIG_MAP.get("LEVEL", 8)
    DIST_WORKERS = CONFIG_MAP.get("DIST_WORKERS", 1)
    LOCAL_RANK = CONFIG_MAP.get("LOCAL_RANK", 0)
    USE_FLASH = CONFIG_MAP.get("USE_FLASH", 0) #not use flash by default
    POOL_FROM_PROMPT = CONFIG_MAP.get("POOL_FROM_PROMPT", 1)
    num_init = int(os.environ.get("NUM_INIT", 20))
    num_local = int(os.environ.get("NUM_LOCAL", 30))
    decoding_mask = CONFIG_MAP.get('decoding_mask', None)
    flex_attention = CONFIG_MAP.get('flex_attention', None)
    curr_ptr = 0
    USE_AWQ = False #not support AWQ
    chat = bool(os.environ.get("CHAT",0))
    GUESS_SIZE = LEVEL - 1
    NOT_SEQ = 0
    CONTINUE_ALL = 0
    TEMP_FOR_GUESS = 0.0
    USE_AWQ = False 
    import random
    assert TEMP_FOR_GUESS == 0
    assert ALWAYS_FWD_ONE == 1
    assert USE_AWQ == False 
    random_color = None

    ############### Init methods

    all_old_tokens = input_ids[0].tolist()
    init_len = len(all_old_tokens)
    order_copy_from_idx = [0]


    def random_set():
        return random.randint(0,self.vocab_size - 1)

    def copy_from():
        return random.choice(all_old_tokens)

    def order_copy_from():
        if order_copy_from_idx[0] >= len(all_old_tokens):
            order_copy_from_idx[0] = 0
        ret = all_old_tokens[order_copy_from_idx[0]]
        order_copy_from_idx[0] = 1 + order_copy_from_idx[0]
        return ret

    def copy_from_last():
        return all_old_tokens[-1]

    set_token = copy_from

    # Why do we need an additional of LEVEL -2 tokens and remove once every filling-up step
    past_tokens = [[set_token() for _ in range(WINDOW_SIZE + LEVEL - 3)]] + [None for _ in range(LEVEL - 2)]
    #past_tokens is the lookahead window. Current we initialize it with random copy from prompts

    ###############end Init methods
    fill_level = 0
    guess_tokens = None
    token_map = {}
    steps = 0
    guess_skip_dist = 0
    PREFIX_LEN = 10
    prefix_tokens = [[] for _ in range(PREFIX_LEN)]
    model_kwargs['_use_cache'] = True
    model_kwargs['attention_mask'] = torch.ones_like(input_ids)
    if POOL_FROM_PROMPT:
        fill_pool_with_prompt(all_old_tokens, token_map, LEVEL, GUESS_SET_SIZE)
        
    if chat:
        init = self.tokenizer.decode(all_old_tokens, skip_special_tokens=True, \
                                   spaces_between_special_tokens=False, clean_up_tokenization_spaces=True,)
        prev = len(init)

    while True:
        # prepare model inputs
        #this only support llama, check compatibility with other models
        past_key_values = model_kwargs.pop("past_key_values", None)
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if past_key_values is None:
            model_inputs["input_ids"] = input_ids
        else:
            model_inputs["input_ids"] = model_inputs["input_ids"][:, -1 - guess_skip_dist:]
            model_inputs["position_ids"] = model_inputs["position_ids"][:, -1 - guess_skip_dist:]
        model_inputs["past_key_values"] = past_key_values

        ori_guess = None
        #set up guess_tokens for verification branch 
        # past_tokens[LEVEL - 2] is None means we are still in warmup stage filling multi-level window
        if past_tokens[LEVEL - 2] is not None and lst_token in token_map and GUESS_SET_SIZE > 0:
            guess_tokens_ = []
            included = {}
            prefix = all_old_tokens[-PREFIX_LEN:]
            for i in range(len(prefix)):
                pre = tuple(prefix[i:]) if i != len(prefix) - 1 else prefix[i]
                if pre in token_map:
                    guesses = token_map[pre]
                    for guess in guesses:
                        if len(guess_tokens_) < GUESS_SET_SIZE and tuple(guess) not in included:
                            guess_tokens_.append(guess)
                            included[tuple(guess)] = 1
                if len(guess_tokens_) >= GUESS_SET_SIZE:
                    break

            guess_tokens = []
            for tok in list(guess_tokens_):
                guess_tokens += list(tok)
            ori_guess = guess_tokens
        elif past_tokens[LEVEL - 2] is not None and GUESS_SET_SIZE > 0:
            guess_tokens_= []
            guess_tokens = []
        else:
            guess_tokens = None
        
        if guess_tokens is not None and len(guess_tokens_) < GUESS_SET_SIZE:
            num_guess = len(guess_tokens_)
            for i in range(num_guess, GUESS_SET_SIZE):
                guess_tokens += [set_token() for _ in range(LEVEL - 1)]
        
        assert return_dict_in_generate == False
        assert len(logits_processor) == 0

        past_tokens_inp = past_tokens

        llookahead = sum(len(past_tokens[i]) for i in range(fill_level + 1))
        if guess_tokens is not None:
            lguess = len(guess_tokens)
        else:
            lguess = 0
        fafo_frame = 1 + llookahead + lguess

        with torch.no_grad(): 
            outputs = self.jforward_multilevel(
                **model_inputs,
                past_tokens=past_tokens_inp,
                guess_tokens=guess_tokens,
                return_dict=True,
                not_seq=NOT_SEQ,
                continue_all=CONTINUE_ALL,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                level=LEVEL,
                WINDOWS_SIZE=WINDOW_SIZE,
                guess_size=GUESS_SIZE,
                fill_level=fill_level,
                dist_workers=DIST_WORKERS,
                decoding_mask=decoding_mask if past_tokens[LEVEL - 2] is not None else None,
                flex_attention=flex_attention,
                la_mask_offset=0,
                local_rank=LOCAL_RANK,
                use_flash=USE_FLASH
            )
        
        steps += 1

        next_token_logits = outputs.out_logits

        # pre-process distribution
        next_tokens_scores = next_token_logits
        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        first_guess = next_tokens.item()
        max_hit = 0 
        hits = [first_guess] + [0] * (GUESS_SIZE - 1)

        new_results = []
        if past_tokens[1] is None: #filling multi-level window, the very first step is different
            assert fill_level == 0
            past_tokens[0] = past_tokens[0][1:] 
            past_tokens[1] = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()

            fill_level += 1
        elif past_tokens[LEVEL - 2] is None: #filling multi-level window
            for level in range(fill_level + 1):
                past_tokens[level] = past_tokens[level][1:] 
            current_past_tokens = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()

            past_tokens[fill_level + 1] = current_past_tokens[1:]
            fill_level += 1
        # FInd n-gram that matches
        else: 
            #match guess tokens 
            if guess_tokens is not None:
                guess_results = torch.argmax(outputs.guess_logits, dim=-1)[0].tolist()
                for eg in range(len(guess_results) // GUESS_SIZE):
                    egx = eg * GUESS_SIZE
                    correct = [first_guess] + guess_results[egx:egx + GUESS_SIZE]
                    myguess = guess_tokens[egx:egx + GUESS_SIZE]
                    gg = 0
                    for gg in range(len(myguess)):
                        if myguess[gg] != correct[gg]:
                            break 
                    if gg > max_hit:
                        max_hit = gg 
                        max_hit_idx = eg 
                        hits[:max_hit + 1] = correct[:max_hit + 1]

            new_results = torch.argmax(outputs.inp_logits, dim=-1)[0].tolist()

            assert len(past_tokens[LEVEL - 2]) == WINDOW_SIZE and len(new_results) == WINDOW_SIZE

            
            update_token_map(token_map, lst_token, past_tokens, new_results, prefix_tokens, LEVEL, WINDOW_SIZE, GUESS_SET_SIZE)


            if ALWAYS_FWD_ONE:
                for level in range(len(prefix_tokens) - 1):
                    prefix_tokens[level] = prefix_tokens[level + 1]
                prefix_tokens[-1] = [lst_token] + past_tokens[0]

                past_tokens[0] = past_tokens[1][1:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][:]

                past_tokens[LEVEL - 2] = new_results             
            else:
                past_tokens[0] = past_tokens[1][1 + max_hit:]
                for level in range(1, LEVEL - 2):
                    past_tokens[level] = past_tokens[level + 1][max_hit:]

                past_tokens[LEVEL - 2] = new_results[max_hit:]


        if max_hit > 0:
            if not ALWAYS_FWD_ONE:
                for level in range(LEVEL - 1):
                    past_tokens[level] = past_tokens[level] + [set_token() for _ in range(max_hit)]

            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat((attention_mask, torch.ones(1, max_hit, device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)
        
        #not support awq
        assert USE_AWQ == False  

        past_key_values = []

        guess_skip_dist = 0
        if past_tokens[LEVEL - 2] is None:
            offset_kv_cache = outputs.step_len-len(guess_tokens)+max_hit_idx * GUESS_SIZE if max_hit > 0 else 0
        else:
            offset_kv_cache = outputs.step_len-len(guess_tokens)-llookahead+max_hit_idx * GUESS_SIZE if max_hit > 0 else 0

        for idx, kv in enumerate(outputs.past_key_values):
            #update kv-cache from verification branch
            if lguess > 0:
                next_token_kv = (kv[0][:,:,0,:], kv[1][:,:,0,:])
                hit_kv = None
                if max_hit > 0:
                    hit_start = 1 + max_hit_idx * GUESS_SIZE
                    hit_end = 1 + max_hit_idx * GUESS_SIZE + max_hit
                    hit_kv = (kv[0][:,:,hit_start:hit_end,:], kv[1][:,:,hit_start:hit_end,:])
                kv[0][:,:,fafo_frame+outputs.kvcache_len-1,:] = next_token_kv[0]
                kv[1][:,:,fafo_frame+outputs.kvcache_len-1,:] = next_token_kv[1]
                if max_hit > 0:
                    kv[0][:,:,fafo_frame+outputs.kvcache_len:fafo_frame+outputs.kvcache_len+max_hit,:] = hit_kv[0]
                    kv[1][:,:,fafo_frame+outputs.kvcache_len:fafo_frame+outputs.kvcache_len+max_hit,:] = hit_kv[1]
                past_key_values.append( (kv[0][:,:,fafo_frame:fafo_frame+outputs.kvcache_len + max_hit,:], kv[1][:,:,fafo_frame:fafo_frame+outputs.kvcache_len + max_hit,:]) )
            else:
                if max_hit > 0:
                    kv[0][:,:,outputs.kvcache_len:outputs.kvcache_len+max_hit,:] = kv[0][:,:,offset_kv_cache:offset_kv_cache+max_hit,:]
                    kv[1][:,:,outputs.kvcache_len:outputs.kvcache_len+max_hit,:] = kv[1][:,:,offset_kv_cache:offset_kv_cache+max_hit,:]
                past_key_values.append( (kv[0][:,:,:outputs.kvcache_len + max_hit,:], kv[1][:,:,:outputs.kvcache_len + max_hit,:]) )

        outputs.past_key_values = past_key_values

        lst_token = hits[max_hit]

        #stopping condition
        for hit_idx in range(max_hit + 1):
            if eos_token_id is not None and hits[hit_idx] == eos_token_id[0]:
                all_old_tokens.append(hits[hit_idx])
                next_tokens = eos_token_id_tensor
                max_hit = hit_idx
                break
            else:
                all_old_tokens.append(hits[hit_idx])
                if POOL_FROM_PROMPT:
                    append_new_generated_pool(all_old_tokens[-LEVEL:], token_map, LEVEL, GUESS_SET_SIZE)


        input_ids = torch.cat([input_ids, torch.tensor(hits[:max_hit + 1], device=next_tokens.device, dtype=next_tokens.dtype).unsqueeze(0)], dim=-1)
        
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self.j_update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True
        
        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if n_new_tokens and input_ids.shape[1] - init_len > n_new_tokens:
            this_peer_finished = True

        if this_peer_finished:
            break
    
    for criteria in stopping_criteria:
        if hasattr(criteria, "max_length"):
            all_old_tokens = all_old_tokens[:criteria.max_length]
            input_ids = input_ids[:,:criteria.max_length]
    if max_length is not None:
        all_old_tokens = all_old_tokens[:init_len + max_length]
        input_ids = input_ids[:][:init_len + max_length]

    if LOCAL_RANK == 0:
        CONFIG_MAP["log"].append([len(all_old_tokens) - init_len, steps, round((len(all_old_tokens) - init_len) / steps, 2)])
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids
