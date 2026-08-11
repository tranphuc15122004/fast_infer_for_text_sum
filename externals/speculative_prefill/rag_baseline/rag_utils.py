"""
    Modified from Yang Zhou's code (https://scholar.google.com/citations?user=W6CZltIAAAAJ&hl=en). 
"""

import re

import faiss
import nltk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

nltk.download('punkt')
nltk.download('punkt_tab')

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load the LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

def process_text(text):
    """
    Processes the input text by adjusting punctuation spacing, normalizing spaces,
    and tokenizing it into sentences.

    Parameters:
    text (str): The input text to be processed.

    Returns:
    list: A list of sentences extracted from the processed text.
    """
    # Adjust punctuation marks
    punctuations = [r'\?', r'\!', r'\.', ',']

    for punc in punctuations:
        # For '?', '!', and '.', add space if not followed by space
        if punc in [r'\?', r'\!', r'\.']:
            text = re.sub(r'{}(?!\s)'.format(punc), '{} '.format(punc.strip('\\')), text)
        # For ',', add space if followed by non-space character
        elif punc == ',':
            text = re.sub(r',(?=\S)', ', ', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize sentences
    sentences = nltk.sent_tokenize(text)

    return sentences

def split_sentences_batch(texts):
    # Split each text in the list based on the special syntax using regular expressions
    all_sentences = []
    
    for text in texts:
        special_sentences = re.findall(r'@<<<.*?>>>@', text)
        
        # Replace the special syntax sentences with placeholders
        placeholder = '__PLACEHOLDER__'
        for sentence in special_sentences:
            text = text.replace(sentence, placeholder)
        
        # Split the remaining text into natural language sentences using NLTK
        # natural_sentences = nltk.sent_tokenize(text)
        natural_sentences = process_text(text)
        
        # Combine the special syntax sentences and natural language sentences
        sentences = []
        for sentence in natural_sentences:
            while placeholder in sentence:
                index = sentence.index(placeholder)
                if special_sentences:
                    sentences.append(special_sentences.pop(0))
                sentence = sentence[:index] + sentence[index + len(placeholder):]
            if sentence.strip():
                sentences.append(sentence.strip())
        
        # Append any remaining special syntax sentences
        sentences.extend(special_sentences)
        # print(sentences)
        # print(len(sentences))
        # print(texts)
        
        all_sentences.append(sentences)
    # print("*"*40) 
    # print(all_sentences) 
    return all_sentences

def retrieve_relevant_sentences(
    contexts, 
    queries, 
    token_budgets, 
): 
    # contexts: list of context strings
    # queries: list of query strings
    # token_budgets: list of token budgets

    # Ensure all inputs are lists of the same length
    assert len(contexts) == len(queries) == len(token_budgets), "Input lists must have the same length."

    # Split the contexts into sentences
    all_sentences = split_sentences_batch(contexts)

    # Flatten all sentences to compute embeddings in batch
    flattened_sentences = [sentence for sentences in all_sentences for sentence in sentences]

    # Create embeddings for all sentences in batch
    sentence_embeddings = model.encode(flattened_sentences, batch_size=512, convert_to_tensor=True)

    # Prepare indices for each context's sentences
    sentence_offsets = [0]
    for sentences in all_sentences[:-1]:
        sentence_offsets.append(sentence_offsets[-1] + len(sentences))

    # Create embeddings for all queries in batch
    query_embeddings = model.encode(queries, batch_size=512, convert_to_tensor=True)

    # For each context-query pair, retrieve relevant sentences
    retrieved_contexts = []
    for i in range(len(contexts)):
        # Get sentences and embeddings for this context
        sentences = all_sentences[i]
        start_idx = sentence_offsets[i]
        end_idx = sentence_offsets[i] + len(sentences)
        context_sentence_embeddings = sentence_embeddings[start_idx:end_idx]

        # Create a FAISS index for this context 
        # print("context_sentence_embeddings.shape {}".format(context_sentence_embeddings.shape)) 
        dimension = context_sentence_embeddings.shape[1] 
        index = faiss.IndexFlatL2(dimension)
        index.add(context_sentence_embeddings.cpu().numpy())

        # Perform similarity search using FAISS
        query_embedding = query_embeddings[i].cpu().numpy().reshape(1, -1)
        num_results = min(token_budgets[i] // 10 + 1, len(sentences))  # Estimate number of sentences
        _, indices = index.search(query_embedding, k=num_results)

        # Select top-k sentences within the token budget
        retrieved_sentences = []
        total_tokens = 0
        for idx in indices[0]:
            sentence = sentences[idx]
            tokens = len(tokenizer.encode(sentence))
            if total_tokens + tokens <= token_budgets[i]:
                retrieved_sentences.append(sentence)
                total_tokens += tokens
            else:
                break 

        # Concatenate the retrieved sentences
        retrieved_context = "\n".join(retrieved_sentences) 
        retrieved_contexts.append(retrieved_context)

    return retrieved_contexts 

def retrieve_query_fn(dataset_name):
    if dataset_name in [
        "narrativeqa", "qasper", 
        "multifieldqa_en", "multifieldqa_zh", 
        "hotpotqa", "2wikimqa", "musique", 
        "dureader", "qmsum", "trec", "samsum", "lsht", 
        "passage_retrieval_en", "passage_retrieval_zh", "repobench-p"]:
        return lambda input: input
    elif dataset_name in ["gov_report", "multi_news", "vcsum"]:
        return lambda input: "Write a summary about the text."
    elif dataset_name in ["passage_count"]:
        return lambda input: "Count the number of distinctive passages."
    elif dataset_name == "triviaqa":
        return lambda input: input[input.find("Question:"):]
    elif dataset_name == "lcc":
        return lambda input: "Predict the next line of code."
    else:
        raise ValueError


if __name__ == "__main__":
    results = retrieve_relevant_sentences(
        ["""Sven Magnus Øen Carlsen[a] (born 30 November 1990) is a Norwegian chess grandmaster. Carlsen is a five-time World Chess Champion, five-time World Rapid Chess Champion, the reigning (shared with Ian Nepomniachtchi) eight time World Blitz Chess Champion and the reigning Chess World Cup Champion. He has held the No. 1 position in the FIDE world chess rankings since 1 July 2011 and trails only Garry Kasparov in time spent as the highest-rated player in the world.[1] His peak rating of 2882 is the highest in history. He also holds the record for the longest unbeaten streak at the elite level in classical chess at 125 games.[2][3]
A chess prodigy, Carlsen finished first in the C group of the Corus chess tournament shortly after he turned 13 and earned the title of grandmaster a few months later. At 15, he won the Norwegian Chess Championship, and later became the youngest ever player to qualify for the Candidates Tournament in 2005.[1] At 17, he finished joint first in the top group of Corus. He surpassed a rating of 2800 at 18, the youngest at the time to do so. In 2010, at 19, he reached No. 1 in the FIDE world rankings, the youngest person ever to do so.
Carlsen became World Chess Champion in 2013 by defeating Viswanathan Anand. He retained his title against Anand the following year and won both the 2014 World Rapid Championship and World Blitz Championship, becoming the first player to hold all three titles simultaneously, a feat which he repeated in 2019 and 2022.[4][5] He defended his classical world title against Sergey Karjakin in 2016, Fabiano Caruana in 2018, and Ian Nepomniachtchi in 2021. Carlsen declined to defend his title in 2023, citing a lack of motivation.[6]
Known for his attacking style as a teenager, Carlsen has since developed into a universal player. He uses a variety of openings to make it harder for opponents to prepare against him and reduce the utility of pre-game computer analysis.[7]""", 
        """Carlsen was born in Tønsberg, Norway, on 30 November 1990[8][9] to Sigrun Øen (1963–2024), a chemical engineer, and Henrik Albert Carlsen, an IT consultant.[10] The family spent one year in Espoo, Finland, and then in Brussels, Belgium, before returning to Norway in 1998, where they lived in Lommedalen, Bærum. They later moved to Haslum.[11] Carlsen showed an aptitude for intellectual challenges at a young age. At two years, he could solve 50-piece jigsaw puzzles; at four, he enjoyed assembling Lego sets with instructions intended for children aged 10–14.[12]"""], 
        ["Who is Magnus?", "Who is Magnus?"], 
        [128, 128]
    )
    print(results)