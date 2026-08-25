import argparse
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch import JumpingKnowledge
from langchain_text_splitters import TokenTextSplitter

from retrieval import *
from utils import *
from prompt_pool import *  
from data_process import get_processed_data, split_corpus_by_doc, eval_data_generation
from graph_construction import mem_retrieval
from training_preparation import integrate_isolated


class ImprovedGAT(nn.Module):
    def __init__(self, in_dim, h_feats, dropout, attn_drop, n_head=4, num_layer=2):
        super(ImprovedGAT, self).__init__()
        self.num_layer = num_layer
        self.n_head = n_head
        self.h_feats = h_feats
        
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(in_dim, h_feats, num_heads=n_head, feat_drop=dropout, attn_drop=attn_drop, 
                   residual=False, activation=None, allow_zero_in_degree=True))
        self.norm_layers.append(nn.LayerNorm(h_feats * n_head))
        self.act_layers.append(nn.GELU())
        
        # Input projection layer for residual connection
        if in_dim != h_feats * n_head:
            self.residual_layers.append(nn.Linear(in_dim, h_feats * n_head))
        else:
            self.residual_layers.append(nn.Identity())
        
        # Subsequent layers
        for _ in range(num_layer - 1):
            self.gat_layers.append(
                GATConv(h_feats * n_head, h_feats, num_heads=n_head, feat_drop=dropout, 
                       attn_drop=attn_drop, residual=False, activation=None, allow_zero_in_degree=True))
            self.norm_layers.append(nn.LayerNorm(h_feats * n_head))
            self.act_layers.append(nn.GELU())
            self.residual_layers.append(nn.Identity())

        self.JKN = JumpingKnowledge(mode='max')
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, in_feat):
        h = in_feat
        hidden_list = []
        
        for l in range(self.num_layer):
            # Save input for residual connection
            residual = self.residual_layers[l](h)
            
            # GAT layer
            h_gat = self.gat_layers[l](g, h).reshape(in_feat.shape[0], -1)
            
            # Residual connection
            h = h_gat + residual
            
            # Layer normalization and activation
            h = self.norm_layers[l](h)
            h = self.act_layers[l](h)
            h = self.dropout(h)
            
            # Collect hidden states for JumpingKnowledge
            hidden_list.append(torch.mean(h.reshape(in_feat.shape[0], self.n_head, -1), dim=1))

        ret = self.JKN(hidden_list)
        return ret


class GoR(nn.Module):
    def __init__(
            self,
            in_dim: int = 768,
            num_hidden: int = 768,
            num_layer: int = 2,
            n_head: int = 4,
            feat_drop: float = 0.2,
            attn_drop: float = 0.1,
    ):
        super(GoR, self).__init__()
        self.encoder = ImprovedGAT(in_dim=in_dim, h_feats=num_hidden, dropout=feat_drop, 
                                  attn_drop=attn_drop, n_head=n_head, num_layer=num_layer)
        
        # Temperature parameter, learnable
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def focal_ranking_loss(self, y_pred, y_true, alpha=2.0, gamma=1.0, padded_value_indicator=-1, reduction="mean"):
        """
        Focal Ranking Loss - Improved ranking loss, focusing on hard samples
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        padded_mask = y_true == padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")
        
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-50, max=50)
        scores_diffs.masked_fill_(torch.isnan(scores_diffs), 0.)
        
        # Focal weight: Focus on hard ranking pairs
        prob = torch.sigmoid(scores_diffs)
        focal_weight = alpha * (1 - prob) ** gamma
        
        scores_diffs_exp = torch.exp(-scores_diffs)
        losses = focal_weight * torch.log(1. + scores_diffs_exp)

        if reduction == "sum":
            loss = torch.sum(losses[padded_pairs_mask])
        elif reduction == "mean":
            loss = torch.mean(losses[padded_pairs_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss

    def improved_contrastive_loss(self, query, positive, negative, temperature=None):
        """
        Improved contrastive learning loss using hard negative mining
        """
        if temperature is None:
            temperature = self.temperature
        
        # Ensure all inputs are 2D tensors [batch_size, hidden_dim]
        if query.dim() == 3:
            query = query.squeeze(1)  # [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        if positive.dim() == 3:
            positive = positive.squeeze(1)  # [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        if negative.dim() == 3:
            # negative: [batch_size, num_neg, hidden_dim] -> Keep 3D
            pass
        elif negative.dim() == 2:
            negative = negative.unsqueeze(1)  # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
            
        # L2 normalization
        query = F.normalize(query, p=2, dim=-1)  # [batch_size, hidden_dim]
        positive = F.normalize(positive, p=2, dim=-1)  # [batch_size, hidden_dim]
        negative = F.normalize(negative, p=2, dim=-1)  # [batch_size, num_neg, hidden_dim]
        
        # Calculate similarity
        pos_sim = torch.sum(query * positive, dim=-1) / temperature  # [batch_size]
        
        # Calculate negative sample similarity
        # query: [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
        # negative: [batch_size, num_neg, hidden_dim]
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        neg_sim = torch.sum(query_expanded * negative, dim=-1) / temperature  # [batch_size, num_neg]
        
        # Hard negative mining: Select the hardest negative samples
        hard_neg_sim, _ = torch.max(neg_sim, dim=-1)  # [batch_size]
        
        # InfoNCE loss with hard negatives
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [batch_size, 1+num_neg]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # Add extra loss for hard negatives
        hard_neg_loss = F.relu(hard_neg_sim - pos_sim + 0.1).mean()
        
        return loss + 0.1 * hard_neg_loss

    def drop_edge(self, g, drop_rate=0.1):
        """
        DropEdge regularization
        """
        if drop_rate <= 0:
            return g
            
        num_edges = g.num_edges()
        if num_edges == 0:
            return g
            
        mask = torch.rand(num_edges, device=g.device) > drop_rate
        
        if mask.sum() == 0:  # Avoid removing all edges
            return g
            
        src, dst = g.edges()
        new_src = src[mask]
        new_dst = dst[mask]
        
        new_g = dgl.graph((new_src, new_dst), num_nodes=g.num_nodes(), device=g.device)
        new_g.ndata.update(g.ndata)
        
        return new_g

    def forward(self, g, x, query_embedding_list, bert_score_list):
        # Ensure the graph has self-loops
        node_indices = torch.arange(g.num_nodes(), device=g.device)
        if not g.has_edges_between(node_indices, node_indices).any():
            g = dgl.add_self_loop(g)
            
        # DropEdge for regularization
        if self.training:
            g = self.drop_edge(g, drop_rate=0.1)
            
        node_rep = self.encoder(g, x)
        node_rep = torch.split(node_rep, g.batch_num_nodes().cpu().numpy().tolist(), dim=0)

        cl_loss_all = 0
        ranking_loss_all = 0
        entropy_all = 0
        
        for ind, (single_rep, query_embedding, bert_score) in enumerate(
                zip(node_rep, query_embedding_list, bert_score_list)):
            bert_score = bert_score.to(x.device)
            q = query_embedding.to(x.device)  # [num_queries, hidden_dim]
            _, bert_sorted_idx = bert_score.sort(dim=-1, descending=True)
            
            # Get positive and negative samples
            p = single_rep[bert_sorted_idx[:, :1]]  # [num_queries, 1, hidden_dim]
            n = single_rep[bert_sorted_idx[:, 1:]]  # [num_queries, num_neg, hidden_dim]
            
            # Construct stronger negative samples
            other_graphs = node_rep[:ind] + node_rep[ind + 1:]
            if len(other_graphs) > 0:
                in_batch_neg_rep = torch.concat(other_graphs, dim=0).unsqueeze(0).repeat(
                    p.shape[0], 1, 1)  # [num_queries, num_other_nodes, hidden_dim]
                n = torch.concat([n, in_batch_neg_rep], dim=1)  # [num_queries, total_neg, hidden_dim]
            
            # Improved contrastive learning loss
            cl_loss = self.improved_contrastive_loss(q, p, n)
            cl_loss_all += cl_loss
            
            # Improved ranking loss
            q_expanded = q.unsqueeze(1)  # [num_queries, 1, hidden_dim]
            p_sim = torch.matmul(q_expanded, p.transpose(1, 2)).squeeze(1)  # [num_queries, 1]
            n_sim = torch.matmul(q_expanded, n.transpose(1, 2)).squeeze(1)  # [num_queries, num_neg]
            ranking_list = torch.concat([p_sim, n_sim], dim=-1)  # [num_queries, 1+num_neg]
            rank_score_prediction = ranking_list[:, :bert_sorted_idx.shape[-1]]
            
            # Label smoothing for ranking
            rank_gt = 1 / torch.arange(1, 1 + rank_score_prediction.shape[-1]).view(1, -1).repeat(
                rank_score_prediction.shape[0], 1).to(x.device)
            rank_gt = rank_gt * 0.9 + 0.1 / rank_score_prediction.shape[-1]  # Label smoothing
            
            ranking_loss_all += self.focal_ranking_loss(rank_score_prediction, rank_gt)
            
            # Entropy regularization
            entropy_all += torch.distributions.Categorical(
                torch.softmax(torch.matmul(q, single_rep.T), dim=-1)).entropy().mean()

        cl_loss_all /= len(query_embedding_list)
        ranking_loss_all /= len(query_embedding_list)
        entropy_all /= len(query_embedding_list)

        return cl_loss_all, ranking_loss_all, entropy_all

def infer_node_embedding(dgl_graph, model_path):
    model = GoR(in_dim=IN_DIM, num_hidden=HIDDEN_DIM, num_layer=NUM_LAYER, n_head=N_HEAD)
    
    # Load model parameters, ignore mismatched parameters
    state_dict = torch.load(model_path)
    model_dict = model.state_dict()
    
    # Filter out mismatched parameters
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict, strict=False)
    
    model = model.encoder
    model.eval()
    model.to(DEVICE)
    dgl_graph = dgl_graph.to(DEVICE)
    dgl_graph = dgl.add_self_loop(dgl_graph)
    node_embedding = model(dgl_graph, dgl_graph.ndata['feat']).detach()
    node_embedding = [i for i in node_embedding]

    return node_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--local_model_path", type=str, default="/path/to/Llama-2-7b-chat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    parser.add_argument("--recall_chunk_num", type=int, default=6)
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    opt = parser.parse_args()
    
    DATASET = opt.dataset
    LOCAL_MODEL_PATH = opt.local_model_path
    SEED = opt.seed
    RETRIEVER = opt.retriever
    CHUNK_SIZE = opt.chunk_size
    CHUNK_OVERLAP = opt.chunk_overlap
    RECALL_CHUNK_NUM = opt.recall_chunk_num
    IN_DIM = opt.in_dim
    HIDDEN_DIM = opt.hidden_dim
    NUM_LAYER = opt.num_layer
    N_HEAD = opt.n_head

    if DATASET == "booksum":
        OPT_TEMPERATURE = 1.0      
        OPT_MAX_LENGTH = 2048      
        OPT_TOP_P = 1.0           
        print(f"{show_time()} BookSum dataset detected - Using optimized parameters: temp={OPT_TEMPERATURE}, max_len={OPT_MAX_LENGTH}, top_p={OPT_TOP_P}")
    elif DATASET == "wcep":
        OPT_TEMPERATURE = 0.1      
        OPT_MAX_LENGTH = 256       
        OPT_TOP_P = 0.9           
        print(f"{show_time()} WCEP dataset detected - Using optimized parameters: temp={OPT_TEMPERATURE}, max_len={OPT_MAX_LENGTH}, top_p={OPT_TOP_P}")
    elif DATASET == "squality":
        OPT_TEMPERATURE = 0.8      
        OPT_MAX_LENGTH = 1792      
        OPT_TOP_P = 0.95          
        print(f"{show_time()} SQuality dataset detected - Using optimized parameters: temp={OPT_TEMPERATURE}, max_len={OPT_MAX_LENGTH}, top_p={OPT_TOP_P}")
    elif DATASET == "govreport":
        OPT_TEMPERATURE = 0.3      
        OPT_MAX_LENGTH = 2560     
        OPT_TOP_P = 0.9           
        print(f"{show_time()} GovReport dataset detected - Using optimized parameters: temp={OPT_TEMPERATURE}, max_len={OPT_MAX_LENGTH}, top_p={OPT_TOP_P}")
    elif DATASET == "qmsum":
        OPT_TEMPERATURE = 0.1
        OPT_MAX_LENGTH = 512
        OPT_TOP_P = 0.95
        print(f"{show_time()} QMSum dataset detected - Using optimized parameters: temp={OPT_TEMPERATURE}, max_len={OPT_MAX_LENGTH}, top_p={OPT_TOP_P}")
    elif DATASET == "narrativeqa":
        OPT_TEMPERATURE = 0.5
        OPT_MAX_LENGTH = 1024
        OPT_TOP_P = 0.9
        print(f"{show_time()} NarrativeQA dataset detected - Using optimized parameters: temp={OPT_TEMPERATURE}, max_len={OPT_MAX_LENGTH}, top_p={OPT_TOP_P}")
    else:
        OPT_TEMPERATURE = opt.temperature
        OPT_MAX_LENGTH = opt.max_length
        OPT_TOP_P = opt.top_p
        print(f"{show_time()} Using command line parameters: temp={OPT_TEMPERATURE}, max_len={OPT_MAX_LENGTH}, top_p={OPT_TOP_P}")

    DO_SAMPLE = opt.do_sample

    set_seed(int(SEED))
    DEVICE = get_device(int(opt.cuda))

    QUERY_TOKENIZER, CTX_TOKENIZER, QUERY_ENCODER, CTX_ENCODER = get_dense_retriever(retriever=RETRIEVER)
    QUERY_ENCODER = QUERY_ENCODER.to(DEVICE)
    CTX_ENCODER = CTX_ENCODER.to(DEVICE)

    TEXT_SPLITTER = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    data = get_processed_data(dataset=DATASET, train=False)
    print("{} #Data: {}".format(show_time(), len(data)))
    data = data[:30]
    check_path("./graph_hierarchical")
    check_path("./result")
    result_recorder = dict()
    
    for ind, sample in enumerate(data):
        print(f"{show_time()} Processing sample {ind+1}/{len(data)}")
        
        all_doc_chunk_list = split_corpus_by_doc(dataset=DATASET, sample=sample, text_splitter=TEXT_SPLITTER)
        all_doc_chunk_list_embedding = get_dense_embedding(all_doc_chunk_list, retriever=RETRIEVER,
                                                           tokenizer=CTX_TOKENIZER,
                                                           model=CTX_ENCODER)
        graph = load_nx(path="./graph_hierarchical/{}_test_hierarchical_graph_{}.graphml".format(DATASET, ind))
        gs, _ = dgl.load_graphs("./graph_hierarchical/{}_test_hierarchical_graph_{}.dgl".format(DATASET, ind))
        dgl_graph = gs[0]
        graph, dgl_graph, = integrate_isolated(graph=graph, dgl_graph=dgl_graph, all_doc_chunk_list=all_doc_chunk_list,
                                               all_doc_chunk_list_embedding=all_doc_chunk_list_embedding)
        check_path("./weights")
        mem_chunk_embedding = infer_node_embedding(dgl_graph=dgl_graph, model_path="./weights/{}.pth".format(DATASET))
        eval_data = eval_data_generation(dataset=DATASET, sample=sample)
        
        for query_ind, test_query in enumerate(eval_data):
            retrieved_chunks, _ = mem_retrieval(mem_chunk_embedding=mem_chunk_embedding,
                                                rag_query=test_query["rag_query"],
                                                graph=graph,
                                                all_doc_chunk_list=all_doc_chunk_list,
                                                all_doc_chunk_list_embedding=all_doc_chunk_list_embedding,
                                                retriever=RETRIEVER,
                                                query_tokenizer=QUERY_TOKENIZER,
                                                query_encoder=QUERY_ENCODER,
                                                recall_chunk_num=RECALL_CHUNK_NUM)
            
            prompt_template = QUERY_PROMPT.get(DATASET, QUERY_PROMPT_NORMAL.get(DATASET))
            
            if prompt_template is None:
                print(f"Warning: No prompt template found for dataset {DATASET}, using generic default.")
                prompt_template = """Refer to the following materials and answer the question.

Materials:
{materials}

Question:
{question}
"""

            # Unified use of \n to splice retrieved text chunks
            materials_str = "\n".join(retrieved_chunks)
            
            final_prompt = prompt_template.format_map({
                "question": test_query["query"],
                "materials": materials_str
            })
            
            response = get_llm_response_via_local(
                prompt=final_prompt,
                MODEL_PATH=LOCAL_MODEL_PATH,
                MAX_LENGTH=OPT_MAX_LENGTH,
                TEMPERATURE=OPT_TEMPERATURE,
                TOP_P=OPT_TOP_P,
                DO_SAMPLE=DO_SAMPLE,
                SEED=SEED)
            
            print(f"{show_time()} Sample {ind+1}, Query {query_ind+1}")
            print(text_wrap("QUERY:"), test_query["query"])
            print(text_wrap("LLM RESPONSE:\n"), response)
            print(text_wrap("GOLDEN ANSWER: {}".format(test_query["summary"])))
            print("-" * 80)
            
            result_recorder[str(ind) + test_query['query']] = {"response": response, "gt": test_query["summary"]}
            
            # Clear GPU cache to avoid memory overflow
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_file = "./result/{}.json".format(DATASET)
    write_to_json(result_recorder, output_file)
    print("{} Evaluation completed. Results saved to {}".format(show_time(), output_file))
