# Resumable
import argparse
import json
import re
import numpy as np
import torch
import networkx as nx
import dgl
from langchain_text_splitters import TokenTextSplitter
import os

from retrieval import *
from utils import *
from prompt_pool import *
from data_process import get_processed_data, split_corpus_by_doc

# Proposition generation prompt template
CLAIM_GENERATE_LOCAL = (
    "From the following text, extract ONE concise, verifiable claim or statement (one sentence, <= 30 words). "
    "Avoid questions. Be factual and specific. Output only the claim without extra text.\n\n"
    "Text: {document}\n\n"
    "Claim:"
)

def rag_retrieval(chunk_list, rag_query, chunk_embedding=None):
    if len(chunk_list) <= RECALL_CHUNK_NUM:
        return chunk_list
    if chunk_embedding is None:
        chunk_embedding = get_dense_embedding(chunk_list, retriever=RETRIEVER, tokenizer=CTX_TOKENIZER,
                                              model=CTX_ENCODER)
    rag_query_embedding = get_dense_embedding([rag_query], retriever=RETRIEVER, tokenizer=QUERY_TOKENIZER,
                                              model=QUERY_ENCODER)
    assert len(rag_query_embedding) == 1
    _, retrieved_text_list = run_dense_retrieval(rag_query_embedding, chunk_embedding, chunk_list,
                                                 chunk_num=RECALL_CHUNK_NUM)

    return retrieved_text_list


def mem_retrieval(mem_chunk_embedding, all_doc_chunk_list, all_doc_chunk_list_embedding, rag_query, graph, retriever,
                  query_tokenizer, query_encoder, recall_chunk_num):
    mem_chunk_list = []
    for node, attrs in graph.nodes(data=True):
        mem_chunk_list.append(node)
    assert len(mem_chunk_embedding) == len(mem_chunk_list), "{}!={}".format(len(mem_chunk_embedding),
                                                                            len(mem_chunk_list))
    mem_chunk_embedding_copy = [i for i in mem_chunk_embedding]
    for chunk, chunk_embedding in zip(all_doc_chunk_list, all_doc_chunk_list_embedding):
        if chunk not in mem_chunk_list:
            mem_chunk_list.append(chunk)
            mem_chunk_embedding_copy.append(chunk_embedding)
    rag_query_embedding = get_dense_embedding([rag_query], retriever=retriever, tokenizer=query_tokenizer,
                                              model=query_encoder)
    mem_chunk_embedding_copy = [i.to(rag_query_embedding[0].device) for i in mem_chunk_embedding_copy]
    assert len(rag_query_embedding) == 1
    assert len(mem_chunk_embedding_copy) == len(mem_chunk_list)
    retrieved_index, retrieved_text_list = run_dense_retrieval(rag_query_embedding, mem_chunk_embedding_copy,
                                                               mem_chunk_list, chunk_num=recall_chunk_num)

    return retrieved_text_list, retrieved_index


def get_node_embedding_list(dgl_graph):
    mem_chunk_embedding = dgl_graph.ndata['feat']
    mem_chunk_embedding = [i for i in mem_chunk_embedding]

    return mem_chunk_embedding


def record_graph_construction(query, support_materials, response, graph, dgl_graph, training_data, answer=None):
    sub_training_data = dict()
    sub_training_data["query"] = query
    if answer:
        sub_training_data["answer"] = answer
    existing_chunks = []
    for node, attrs in graph.nodes(data=True):
        existing_chunks.append(node)
    non_dup_chunks = []
    if response not in existing_chunks:
        non_dup_chunks.append(response)
        graph.add_node(
            response,
        )
        existing_chunks.append(response)
    for chunk in support_materials:
        if chunk not in existing_chunks:
            non_dup_chunks.append(chunk)
            graph.add_node(
                chunk,
            )
            existing_chunks.append(chunk)
    chunk_id_map = dict()
    for chunk_id, chunk in enumerate(existing_chunks):
        chunk_id_map[chunk] = chunk_id
    if len(non_dup_chunks) != 0:
        new_node_embedding = get_dense_embedding(non_dup_chunks, retriever=RETRIEVER, tokenizer=CTX_TOKENIZER,
                                                 model=CTX_ENCODER)
        dgl_graph.add_nodes(num=len(non_dup_chunks), data={'feat': torch.vstack(new_node_embedding).cpu()})
    sub_training_data["response"] = [chunk_id_map[response]]
    sub_training_data["raw"] = []
    for chunk in support_materials:
        sub_training_data["raw"].append(chunk_id_map[chunk])
        if not graph.has_edge(chunk, response):
            graph.add_edge(
                chunk,
                response,
                weight=1
            )
        if not dgl_graph.has_edges_between(chunk_id_map[chunk], chunk_id_map[response]):
            dgl_graph.add_edges(chunk_id_map[chunk],
                                chunk_id_map[response],
                                data={'w': torch.ones(1, 1)})

    training_data.append(sub_training_data)

    return graph, dgl_graph, training_data


def is_valid_question(question):
    """Helper function to validate question quality"""
    if not question or len(question.strip()) == 0:
        return False
        
    words = question.split()
    
    # Length check
    if not (5 <= len(words) <= 50):
        return False
    
    # Must contain a question mark
    if '?' not in question:
        return False
    
    # Must contain question words or question structure
    question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 
                     'can', 'could', 'would', 'should', 'do', 'does', 'did',
                     'is', 'are', 'was', 'were', 'will', 'shall'}
    
    if not any(word.lower() in question_words for word in words[:8]):
        return False
    
    # Avoid excessive repeated words
    unique_words = set(word.lower() for word in words)
    if len(unique_words) / len(words) < 0.6:
        return False
    
    # Avoid obvious error indicators
    error_indicators = ['error', 'failed', 'null', 'undefined', 'json', 'api']
    if any(indicator in question.lower() for indicator in error_indicators):
        return False
    
    return True


def is_valid_statement(statement):
    """Helper function to validate proposition (statement) quality"""
    if not statement or len(statement.strip()) == 0:
        return False
        
    words = statement.split()
    
    # Length check
    if not (5 <= len(words) <= 50):
        return False
    
    # Should not contain a question mark (statement, not a question)
    if '?' in statement:
        return False
    
    # Avoid excessive repeated words
    unique_words = set(word.lower() for word in words)
    if len(unique_words) / len(words) < 0.6:
        return False
    
    # Avoid obvious error indicators
    error_indicators = ['error', 'failed', 'null', 'undefined', 'json', 'api']
    if any(indicator in statement.lower() for indicator in error_indicators):
        return False
    
    # Simple check for verbs (propositions should have predicates)
    common_verbs = {'is', 'are', 'was', 'were', 'has', 'have', 'includes', 'uses', 
                   'contains', 'leads', 'causes', 'shows', 'demonstrates', 'provides',
                   'enables', 'allows', 'requires', 'involves', 'represents'}
    
    if not any(word.lower() in common_verbs for word in words):
        # If no common verbs, at least sufficient word length
        if len(words) < 8:
            return False
    
    return True

def llm_judge_claim_quality(claim: str, original_text: str) -> dict:
    """Use LLM as a judge to assess proposition quality"""
    
    judge_prompt = f"""As an expert evaluator, please assess the quality of this extracted claim based on the original text.

Original Text:
{original_text}

Extracted Claim:
{claim}

Please evaluate on these criteria (score 1-5 for each):

1. **Factual Accuracy**: Is the claim factually consistent with the original text?
2. **Verifiability**: Can this claim be verified or fact-checked?
3. **Completeness**: Does the claim capture important information from the text?
4. **Clarity**: Is the claim clear and unambiguous?
5. **Specificity**: Is the claim specific enough to be meaningful?

Provide scores and a brief explanation. Format your response as JSON:
{{
    "factual_accuracy": <score 1-5>,
    "verifiability": <score 1-5>, 
    "completeness": <score 1-5>,
    "clarity": <score 1-5>,
    "specificity": <score 1-5>,
    "overall_score": <average score>,
    "explanation": "<brief explanation>",
    "accept": <true/false for overall_score >= 3.0>
}}"""
    
    try:
        response = get_llm_response_via_api(prompt=judge_prompt, 
                                           LLM_MODEL=LLM_MODEL, 
                                           TAU=0.1,  # Low temperature for consistency
                                           SEED=SEED)
        judgment = parse_llm_judgment(response)
        return judgment
    except Exception as e:
        print(f"LLM judgment failed: {e}")
        return {"accept": True, "overall_score": 3.0, "explanation": "Judgment failed, defaulting to accept"}

def parse_llm_judgment(response: str) -> dict:
    """Parse LLM judge's JSON response"""
    try:
        # Try to parse JSON directly
        if response.strip().startswith('{'):
            judgment = json.loads(response.strip())
        else:
            # Extract JSON part from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                judgment = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
        
        # Validate required fields
        required_fields = ['factual_accuracy', 'verifiability', 'completeness', 'clarity', 'specificity']
        for field in required_fields:
            if field not in judgment:
                judgment[field] = 3  # Default medium score
        
        # Calculate overall score and acceptance status
        scores = [judgment[field] for field in required_fields]
        judgment['overall_score'] = sum(scores) / len(scores)
        judgment['accept'] = judgment['overall_score'] >= 3.0
        
        if 'explanation' not in judgment:
            judgment['explanation'] = f"Average score: {judgment['overall_score']:.1f}"
            
        return judgment
        
    except Exception as e:
        print(f"Failed to parse LLM judgment: {e}")
        print(f"Raw response: {response[:200]}...")
        # Return default judgment
        return {
            "factual_accuracy": 3,
            "verifiability": 3, 
            "completeness": 3,
            "clarity": 3,
            "specificity": 3,
            "overall_score": 3.0,
            "explanation": f"Parse failed: {str(e)[:100]}",
            "accept": True
        }

def llm_judge_question_quality(question: str, original_text: str) -> dict:
    """Use LLM as a judge to assess question quality"""
    
    judge_prompt = f"""As an expert evaluator, please assess the quality of this generated question based on the original text.

Original Text:
{original_text}

Generated Question:
{question}

Please evaluate on these criteria (score 1-5 for each):

1. **Relevance**: Is the question directly related to the original text?
2. **Answerability**: Can this question be answered using the original text?
3. **Clarity**: Is the question clear and well-formed?
4. **Specificity**: Is the question specific enough to elicit a meaningful answer?
5. **Complexity**: Does the question require understanding rather than simple recall?

Provide scores and a brief explanation. Format your response as JSON:
{{
    "relevance": <score 1-5>,
    "answerability": <score 1-5>, 
    "clarity": <score 1-5>,
    "specificity": <score 1-5>,
    "complexity": <score 1-5>,
    "overall_score": <average score>,
    "explanation": "<brief explanation>",
    "accept": <true/false for overall_score >= 3.0>
}}"""
    
    try:
        response = get_llm_response_via_api(prompt=judge_prompt, 
                                           LLM_MODEL=LLM_MODEL, 
                                           TAU=0.1,  # Low temperature for consistency
                                           SEED=SEED)
        judgment = parse_llm_question_judgment(response)
        return judgment
    except Exception as e:
        print(f"LLM question judgment failed: {e}")
        return {"accept": True, "overall_score": 3.0, "explanation": "Judgment failed, defaulting to accept"}

def parse_llm_question_judgment(response: str) -> dict:
    """Parse LLM judge's JSON response for questions"""
    try:
        # Try to parse JSON directly
        if response.strip().startswith('{'):
            judgment = json.loads(response.strip())
        else:
            # Extract JSON part from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                judgment = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
        
        # Validate required fields
        required_fields = ['relevance', 'answerability', 'clarity', 'specificity', 'complexity']
        for field in required_fields:
            if field not in judgment:
                judgment[field] = 3  # Default medium score
        
        # Calculate overall score and acceptance status
        scores = [judgment[field] for field in required_fields]
        judgment['overall_score'] = sum(scores) / len(scores)
        judgment['accept'] = judgment['overall_score'] >= 3.0
        
        if 'explanation' not in judgment:
            judgment['explanation'] = f"Average score: {judgment['overall_score']:.1f}"
            
        return judgment
        
    except Exception as e:
        print(f"Failed to parse LLM question judgment: {e}")
        # Return default judgment
        return {
            "relevance": 3,
            "answerability": 3, 
            "clarity": 3,
            "specificity": 3,
            "complexity": 3,
            "overall_score": 3.0,
            "explanation": f"Parse failed: {str(e)[:100]}",
            "accept": True
        }

def llm2query(prompt, tau=0.5):
    try:
        content = get_llm_response_via_api(prompt=prompt,
                                           LLM_MODEL=LLM_MODEL,
                                           TAU=tau,
                                           SEED=SEED)
    except AttributeError as e:
        print(f"Warning: API response format error: {e}")
        return []
    except Exception as e:
        print(f"Warning: API call failed: {e}")
        return []
    
    if not content or not isinstance(content, str):
        print(f"Warning: Invalid API response content: {content}")
        return []
    
    # 1. First try to parse JSON format response
    try:
        json_data = json.loads(content)
        if isinstance(json_data, dict) and 'choices' in json_data:
            # Handle OpenAI API format response
            if json_data['choices'] and 'message' in json_data['choices'][0]:
                actual_content = json_data['choices'][0]['message'].get('content', '')
                if actual_content:
                    content = actual_content
    except (json.JSONDecodeError, KeyError, IndexError):
        # If not JSON format, continue using original content
        pass
    
    # 2. Clean and extract questions
    # Remove possible JSON residue
    content = re.sub(r'^.*?"content":"', '', content)
    content = re.sub(r'","refusal.*$', '', content)
    
    # Split by line
    lines = content.split("\n")
    
    # 3. Find valid question lines
    valid_questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove leading non-letter characters (numbers, symbols, etc.)
        cleaned_line = re.sub(r'^[^\w\("]*', '', line)
        
        # Basic quality check
        if (len(cleaned_line.split()) >= 5 and  # At least 5 words
            len(cleaned_line) <= 500 and        # No more than 500 characters
            '?' in cleaned_line and             # Contains question mark
            not cleaned_line.startswith('id') and  # Not a JSON field
            not cleaned_line.startswith('{') and   # Not JSON start
            not cleaned_line.startswith('"')):    # Not quote start
            
            # Ensure ends with question mark
            if not cleaned_line.endswith('?'):
                if '?' in cleaned_line:
                    cleaned_line = cleaned_line[:cleaned_line.rfind('?')+1]
                else:
                    continue
                    
            valid_questions.append(cleaned_line)
    
    # 4. Return best question
    if valid_questions:
        # Prefer the last valid question (usually the final output of LLM)
        return [valid_questions[-1]]
    
    # 5. If no valid question found, try looser extraction
    # Find sentences containing question words
    question_patterns = [
        r'[A-Z][^.!?]*\b(what|how|why|when|where|who|which|can|could|would|should|do|does|did|is|are|was|were)\b[^.!?]*\?',
        r'\b(what|how|why|when|where|who|which|can|could|would|should|do|does|did|is|are|was|were)\b[^.!?]*\?'
    ]
    
    for pattern in question_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            # Take the first matching complete sentence
            match_start = content.lower().find(matches[0].lower())
            if match_start != -1:
                # Find question mark position
                question_end = content.find('?', match_start)
                if question_end != -1:
                    question = content[match_start:question_end+1].strip()
                    if len(question.split()) >= 5:
                        return [question]
    
    print(f"Warning: Failed to extract valid question from: {content[:200]}...")
    return []

def llm2claim(prompt, tau=0.3):
    """Extract proposition (statement) from LLM response"""
    try:
        content = get_llm_response_via_api(prompt=prompt,
                                           LLM_MODEL=LLM_MODEL,
                                           TAU=tau,
                                           SEED=SEED)
    except AttributeError as e:
        print(f"Warning: API response format error: {e}")
        return []
    except Exception as e:
        print(f"Warning: API call failed: {e}")
        return []
    
    if not content or not isinstance(content, str):
        print(f"Warning: Invalid API response content: {content}")
        return []
    
    # 1. First try to parse JSON format response
    try:
        json_data = json.loads(content)
        if isinstance(json_data, dict) and 'choices' in json_data:
            # Handle OpenAI API format response
            if json_data['choices'] and 'message' in json_data['choices'][0]:
                actual_content = json_data['choices'][0]['message'].get('content', '')
                if actual_content:
                    content = actual_content
    except (json.JSONDecodeError, KeyError, IndexError):
        # If not JSON format, continue using original content
        pass
    
    # 2. Clean and extract proposition
    # Remove possible JSON residue
    content = re.sub(r'^.*?"content":"', '', content)
    content = re.sub(r'","refusal.*$', '', content)
    
    # Split by line
    lines = content.split("\n")
    
    # 3. Find valid proposition lines
    valid_claims = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove leading non-letter characters (numbers, symbols, etc.)
        cleaned_line = re.sub(r'^[^\w\("]*', '', line)
        
        # Basic quality check
        if (len(cleaned_line.split()) >= 5 and  # At least 5 words
            len(cleaned_line) <= 500 and        # No more than 500 characters
            '?' not in cleaned_line and         # No question mark (statement)
            not cleaned_line.startswith('id') and  # Not a JSON field
            not cleaned_line.startswith('{') and   # Not JSON start
            not cleaned_line.startswith('"')):    # Not quote start
            
            # Ensure ends with period or no punctuation
            if not cleaned_line.endswith('.') and not cleaned_line.endswith(','):
                cleaned_line += '.'
                    
            valid_claims.append(cleaned_line)
    
    # 4. Return best proposition
    if valid_claims:
        # Prefer the last valid proposition (usually the final output of LLM)
        return [valid_claims[-1]]
    
    # 5. If no valid proposition found, try to extract from overall content
    content_cleaned = content.strip()
    if is_valid_statement(content_cleaned):
        return [content_cleaned]
    
    print(f"Warning: Failed to extract valid claim from: {content[:200]}...")
    return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--train', action='store_true')
    # parser.add_argument("--llm_model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--llm_model", type=str, default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0)
    parser.add_argument("--query_tau", type=float, default=0.5)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    parser.add_argument("--recall_chunk_num", type=int, default=6)
    parser.add_argument("--query_num", type=int, default=30)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--construct_mode", type=str, choices=["qa", "claim"], default="qa",
                        help="qa: QA-driven mode; claim: Proposition-evidence driven mode")
    parser.add_argument("--enable_llm_judge", action="store_true", 
                        help="Enable LLM-as-a-Judge deep quality assessment")
    parser.add_argument("--judge_sample_ratio", type=float, default=0.3, 
                        help="Random sampling ratio for LLM judge verification (0.0-1.0)")
    parser.add_argument("--judge_threshold", type=float, default=3.0,
                        help="Minimum score threshold accepted by LLM judge")
    opt = parser.parse_args()
    DATASET = opt.dataset
    TRAIN = opt.train
    LLM_MODEL = opt.llm_model
    SEED = opt.seed
    TAU = opt.tau
    QUERY_TAU = opt.query_tau
    RETRIEVER = opt.retriever
    CHUNK_SIZE = opt.chunk_size
    CHUNK_OVERLAP = opt.chunk_overlap
    RECALL_CHUNK_NUM = opt.recall_chunk_num
    QUERY_NUM = opt.query_num
    START_INDEX = opt.start_index
    CONSTRUCT_MODE = opt.construct_mode
    ENABLE_LLM_JUDGE = opt.enable_llm_judge
    JUDGE_SAMPLE_RATIO = opt.judge_sample_ratio
    JUDGE_THRESHOLD = opt.judge_threshold

    set_seed(int(SEED))
    DEVICE = get_device(int(opt.cuda))

    QUERY_TOKENIZER, CTX_TOKENIZER, QUERY_ENCODER, CTX_ENCODER = get_dense_retriever(retriever=RETRIEVER)
    QUERY_ENCODER = QUERY_ENCODER.to(DEVICE)
    CTX_ENCODER = CTX_ENCODER.to(DEVICE)

    TEXT_SPLITTER = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    data = get_processed_data(dataset=DATASET, train=TRAIN)
    print("{} #Data: {}".format(show_time(), len(data)))
    MAX_NUM = 400 if TRAIN else 30
    data = data[:MAX_NUM]
    check_path("./graph")
    
    # Process data starting from specified index
    for ind, sample in enumerate(data[START_INDEX:], START_INDEX):
        # Check if corresponding file already exists, if so skip
        file_exists = False
        if TRAIN:
            if (os.path.exists("./graph/{}_graph_{}.graphml".format(DATASET, ind)) and
                os.path.exists("./graph/{}_graph_{}.dgl".format(DATASET, ind)) and
                os.path.exists("./graph/{}_training_data_{}.pkl".format(DATASET, ind))):
                file_exists = True
        else:
            if (os.path.exists("./graph/{}_test_graph_{}.graphml".format(DATASET, ind)) and
                os.path.exists("./graph/{}_test_graph_{}.dgl".format(DATASET, ind))):
                file_exists = True
        
        if file_exists:
            print("{} File already exists, skipping index {}".format(show_time(), ind))
            continue
            
        # Due to budget constraints, we randomly select at most 400 samples for training and 30 samples for evaluation.
        # You can optionally create a dev set for hyper-parameter tuning
        all_doc_chunk_list = split_corpus_by_doc(dataset=DATASET, sample=sample, text_splitter=TEXT_SPLITTER)
        all_doc_chunk_list_embedding = get_dense_embedding(all_doc_chunk_list, retriever=RETRIEVER,
                                                           tokenizer=CTX_TOKENIZER,
                                                           model=CTX_ENCODER)
        graph = nx.Graph()
        dgl_graph = dgl.graph(([], []), num_nodes=0)
        training_data = []
        
        # Adjust variable names and generation logic according to the mode
        user_units = []  # Store questions in qa mode, propositions in claim mode
        user_answer = []
        failed_attempts = 0
        max_failed_attempts = QUERY_NUM * 3  # Maximum 3 times the number of attempts
        llm_judge_stats = {"total_generated": 0, "rule_passed": 0, "llm_judged": 0, "llm_accepted": 0}
        
        mode_name = "Question" if CONSTRUCT_MODE == "qa" else "Proposition"
        
        while len(user_units) < QUERY_NUM and failed_attempts < max_failed_attempts:
            unsup_answer = np.random.choice(all_doc_chunk_list, size=1, replace=False)[0].split()
            unsup_answer = " ".join(unsup_answer)
            
            if CONSTRUCT_MODE == "qa":
                gen_list = llm2query(prompt=QUERY_GENERATE.format_map({"document": unsup_answer}), tau=QUERY_TAU)
            else:  # claim mode
                gen_list = llm2claim(prompt=CLAIM_GENERATE_LOCAL.format(document=unsup_answer), tau=QUERY_TAU)
            
            if not gen_list:
                failed_attempts += 1
                print(f"{show_time()} Failed to generate {mode_name}, attempt {failed_attempts}")
                continue
                
            gen_unit = gen_list[0]
            llm_judge_stats["total_generated"] += 1
            
            # First layer validation: Rule-based validation
            if CONSTRUCT_MODE == "qa":
                is_valid_rule = is_valid_question(gen_unit)
            else:  # claim mode
                is_valid_rule = is_valid_statement(gen_unit)
                
            if not is_valid_rule:
                failed_attempts += 1
                print(f"{show_time()} Invalid {mode_name} by rules: {gen_unit}")
                continue
            
            llm_judge_stats["rule_passed"] += 1
            
            # Second layer validation: LLM-as-a-Judge (optional)
            should_judge = ENABLE_LLM_JUDGE and (JUDGE_SAMPLE_RATIO >= 1.0 or np.random.random() < JUDGE_SAMPLE_RATIO)
            
            if should_judge:
                llm_judge_stats["llm_judged"] += 1
                if CONSTRUCT_MODE == "qa":
                    judgment = llm_judge_question_quality(gen_unit, unsup_answer)
                else:  # claim mode
                    judgment = llm_judge_claim_quality(gen_unit, unsup_answer)
                
                if not judgment["accept"] or judgment["overall_score"] < JUDGE_THRESHOLD:
                    failed_attempts += 1
                    print(f"{show_time()} {mode_name} rejected by LLM judge (score: {judgment['overall_score']:.2f}): {gen_unit}")
                    print(f"  Reason: {judgment['explanation']}")
                    continue
                else:
                    llm_judge_stats["llm_accepted"] += 1
                    print(f"{show_time()} {mode_name} accepted by LLM judge (score: {judgment['overall_score']:.2f})")
            
            if gen_unit not in user_units:
                user_units.append(gen_unit)
                user_answer.append(unsup_answer)
                print("{} Generate {} {}/{}:\n{}".format(show_time(), mode_name, len(user_units), QUERY_NUM, gen_unit))
                failed_attempts = 0  # Reset failure count
            else:
                failed_attempts += 1
                print(f"{show_time()} Duplicate {mode_name}: {gen_unit}")
        
        # Output statistics
        print(f"\n{show_time()} Generation Statistics:")
        print(f"  Total generated: {llm_judge_stats['total_generated']}")
        print(f"  Passed rule validation: {llm_judge_stats['rule_passed']}")
        if ENABLE_LLM_JUDGE:
            print(f"  LLM judged: {llm_judge_stats['llm_judged']}")
            print(f"  LLM accepted: {llm_judge_stats['llm_accepted']}")
            if llm_judge_stats['llm_judged'] > 0:
                acceptance_rate = llm_judge_stats['llm_accepted'] / llm_judge_stats['llm_judged']
                print(f"  LLM acceptance rate: {acceptance_rate:.2%}")
        
        if len(user_units) < QUERY_NUM:
            print(f"{show_time()} Warning: Only generated {len(user_units)} {mode_name}s out of {QUERY_NUM} requested")
            
        # Graph Construction
        for uid, user_unit in enumerate(user_units):
            # user_unit is question in qa mode, proposition in claim mode
            if graph.number_of_nodes() == 0:
                retrieved_chunks = rag_retrieval(chunk_list=all_doc_chunk_list, rag_query=user_unit,
                                                 chunk_embedding=all_doc_chunk_list_embedding)
            else:
                mem_chunk_embedding = get_node_embedding_list(dgl_graph=dgl_graph)
                retrieved_chunks, _ = mem_retrieval(mem_chunk_embedding=mem_chunk_embedding, rag_query=user_unit,
                                                    graph=graph, all_doc_chunk_list=all_doc_chunk_list,
                                                    all_doc_chunk_list_embedding=all_doc_chunk_list_embedding,
                                                    retriever=RETRIEVER, query_tokenizer=QUERY_TOKENIZER,
                                                    query_encoder=QUERY_ENCODER, recall_chunk_num=RECALL_CHUNK_NUM)
            
            if CONSTRUCT_MODE == "qa":
                # Original QA mode: Generate answer
                response = get_llm_response_via_api(prompt=QUERY_PROMPT[DATASET].format_map({"question": user_unit,
                                                                                             "materials": "\n\n".join(
                                                                                                 retrieved_chunks)}),
                                                    LLM_MODEL=LLM_MODEL,
                                                    TAU=TAU,
                                                    SEED=SEED)
            else:  # claim mode
                # Proposition mode: Proposition itself is the response node, no need to generate answer
                response = user_unit
            
            graph, dgl_graph, training_data = record_graph_construction(query=user_unit,
                                                                        support_materials=retrieved_chunks,
                                                                        response=response, graph=graph,
                                                                        dgl_graph=dgl_graph,
                                                                        training_data=training_data,
                                                                        answer=user_answer[uid])
            print("{} Graph Construction: {}/{}".format(show_time(), uid + 1, len(user_units)))
            print(dgl_graph)
        # Save
        if TRAIN:
            store_nx(nx_obj=graph, path="./graph/{}_graph_{}.graphml".format(DATASET, ind))
            dgl.save_graphs(filename="./graph/{}_graph_{}.dgl".format(DATASET, ind), g_list=[dgl_graph])
            write_to_pkl(data=training_data, output_file="./graph/{}_training_data_{}.pkl".format(DATASET, ind))
        else:
            store_nx(nx_obj=graph, path="./graph/{}_test_graph_{}.graphml".format(DATASET, ind))
            dgl.save_graphs(filename="./graph/{}_test_graph_{}.dgl".format(DATASET, ind), g_list=[dgl_graph])