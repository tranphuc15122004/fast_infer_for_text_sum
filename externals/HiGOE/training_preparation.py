import argparse
import os 
import dgl
from tqdm import tqdm
from langchain_text_splitters import TokenTextSplitter

from retrieval import *
from utils import *
from sum_eval import bert_score_eval
from data_process import get_processed_data, split_corpus_by_doc


def training_data_generation(graph, training_data):
    queries = [i["query"] for i in training_data]
    queries_embedding = get_dense_embedding(queries, retriever=RETRIEVER, tokenizer=QUERY_TOKENIZER,
                                            model=QUERY_ENCODER)
    queries_embedding = [i.cpu() for i in queries_embedding]
    bert_score = None
    if "answer" in training_data[0]:
        responses = []
        for node, attrs in graph.nodes(data=True):
            responses.append(node)
        answers = []
        for i in training_data:
            answers.extend([i["answer"]] * len(responses))
        responses = responses * len(training_data)
        _, _, bert_score = bert_score_eval(generate_response=responses, ground_truth=answers, device=DEVICE)
        bert_score = np.array(bert_score).reshape((len(training_data), -1))

    return queries_embedding, bert_score


def integrate_isolated(graph, dgl_graph, all_doc_chunk_list, all_doc_chunk_list_embedding):
    raw_chunk = []
    for node, attrs in graph.nodes(data=True):
        raw_chunk.append(node)
    non_dup_chunk = []
    non_dup_chunk_embedding = []
    for chunk, chunk_embedding in zip(all_doc_chunk_list, all_doc_chunk_list_embedding):
        if chunk not in raw_chunk:
            graph.add_node(chunk)
            raw_chunk.append(chunk)
            non_dup_chunk.append(chunk)
            non_dup_chunk_embedding.append(chunk_embedding)

    if len(non_dup_chunk) != 0:
        dgl_graph.add_nodes(num=len(non_dup_chunk), data={'feat': torch.vstack(non_dup_chunk_embedding).cpu()})

    return graph, dgl_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    parser.add_argument("--recall_chunk_num", type=int, default=6)
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="checkpoint interval") 
    opt = parser.parse_args()
    DATASET = opt.dataset
    SEED = opt.seed
    RETRIEVER = opt.retriever
    CHUNK_SIZE = opt.chunk_size
    CHUNK_OVERLAP = opt.chunk_overlap
    RECALL_CHUNK_NUM = opt.recall_chunk_num
    CHECKPOINT_INTERVAL = opt.checkpoint_interval 

    set_seed(int(SEED))
    DEVICE = get_device(int(opt.cuda))
    
    check_path("./training_data")
    CHECKPOINT_FILE = f"./training_data/{DATASET}_checkpoint.pkl"

    QUERY_TOKENIZER, CTX_TOKENIZER, QUERY_ENCODER, CTX_ENCODER = get_dense_retriever(retriever=RETRIEVER)
    QUERY_ENCODER = QUERY_ENCODER.to(DEVICE)
    CTX_ENCODER = CTX_ENCODER.to(DEVICE)

    TEXT_SPLITTER = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    data = get_processed_data(dataset=DATASET, train=True)
    print("{} #Data: {}".format(show_time(), len(data)))
    data = data[:400]
    
    start_index = 0
    gs_list = []
    query_embedding_list = []
    bert_score_list = []

    if os.path.exists(CHECKPOINT_FILE):
        print(f"{show_time()} find checkpoint file. Loading...")
        checkpoint = read_from_pkl(CHECKPOINT_FILE)
        start_index = checkpoint['last_index'] + 1
        gs_list = checkpoint['gs_list']
        query_embedding_list = checkpoint['query_embedding_list']
        bert_score_list = checkpoint['bert_score_list']
    else:
        print(f"{show_time()} No checkpoint file found. Start from scratch.")

    for ind in tqdm(range(start_index, len(data)), initial=start_index, total=len(data), desc="Processing data"):
        sample = data[ind]
        all_doc_chunk_list = split_corpus_by_doc(dataset=DATASET, sample=sample, text_splitter=TEXT_SPLITTER)
        all_doc_chunk_list_embedding = get_dense_embedding(all_doc_chunk_list, retriever=RETRIEVER,
                                                           tokenizer=CTX_TOKENIZER,
                                                           model=CTX_ENCODER)
        try:
            graph = load_nx(path="./graph_hierarchical/{}_hierarchical_graph_{}.graphml".format(DATASET, ind))
            gs, _ = dgl.load_graphs("./graph_hierarchical/{}_hierarchical_graph_{}.dgl".format(DATASET, ind))
            dgl_graph = gs[0]
            training_data = read_from_pkl(output_file="./graph_hierarchical/{}_hierarchical_training_data_{}.pkl".format(DATASET, ind))
        except Exception as e:
            print(f"Failed to load file with index {ind}: {e}, skipping this file.")
            continue
            
        graph, dgl_graph = integrate_isolated(graph=graph, dgl_graph=dgl_graph, all_doc_chunk_list=all_doc_chunk_list,
                                              all_doc_chunk_list_embedding=all_doc_chunk_list_embedding)
        queries_embedding, bert_score = training_data_generation(graph=graph, training_data=training_data)
        gs_list.append(dgl_graph)
        query_embedding_list.append(queries_embedding)
        bert_score_list.append(bert_score)
        
        if (ind + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_data = {
                'last_index': ind,
                'gs_list': gs_list,
                'query_embedding_list': query_embedding_list,
                'bert_score_list': bert_score_list
            }
            write_to_pkl(data=checkpoint_data, output_file=CHECKPOINT_FILE)
            
    print(f"\n{show_time()} all data processed. Saving results...")
    dgl.save_graphs("./training_data/{}_gs.dgl".format(DATASET), gs_list)
    write_to_pkl(data=query_embedding_list, output_file="./training_data/{}_qe.pkl".format(DATASET))
    write_to_pkl(data=bert_score_list, output_file="./training_data/{}_bs.pkl".format(DATASET))
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"{show_time()} Final result has been saved, checkpoint file {CHECKPOINT_FILE} has been deleted.")

    print(f"{show_time()} Done.")