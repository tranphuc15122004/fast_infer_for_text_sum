import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch import JumpingKnowledge

from utils import *


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


def train_gor(train_dataloader):
    model = GoR(in_dim=IN_DIM, num_hidden=HIDDEN_DIM, num_layer=NUM_LAYER, n_head=N_HEAD, feat_drop=DROPOUT)
    model.to(DEVICE)
    
    num_steps = len(train_dataloader) * MAX_EPOCH
    
    # Improved optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=num_steps, 
        pct_start=0.1, anneal_strategy='cos'
    )
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    for e in range(MAX_EPOCH):
        model.train()
        epoch_loss = 0
        entropy_loss = 0
        
        for batch_id, (g, query_embedding_l, bert_score_l) in enumerate(train_dataloader):
            g = g.to(DEVICE)
            
            cl_loss, ranking_loss, entropy = model(g, g.ndata['feat'], query_embedding_l, bert_score_l)
            
            # Dynamic weight adjustment
            epoch_progress = e / MAX_EPOCH
            ranking_weight = COE * (1 + 0.5 * epoch_progress)  # Increase ranking loss weight later
            
            loss = cl_loss + ranking_weight * ranking_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.detach().cpu()
            entropy_loss += entropy.detach().cpu()
            
        print('{} In epoch {}, lr: {:.5f}, loss: {:.4f}, entropy: {:.4f}, temp: {:.4f}'.format(
            show_time(), e, optimizer.param_groups[0]['lr'],
            float(epoch_loss / len(train_dataloader)), 
            float(entropy_loss / len(train_dataloader)),
            model.temperature.item()))

    check_path("./weights")
    torch.save(model.state_dict(), "./weights/{}.pth".format(DATASET))


class GraphDataloader(dgl.data.DGLDataset):
    def __init__(self, query_embedding_list, gs_list, bert_score_list):
        self.query_embedding_list = query_embedding_list
        self.gs_list = gs_list
        self.bert_score_list = bert_score_list
        super(GraphDataloader, self).__init__(name="GraphDataloader")

    def process(self):
        pass

    def __getitem__(self, index):
        return self.gs_list[index], self.query_embedding_list[index], self.bert_score_list[index]

    def __len__(self):
        return int(len(self.gs_list))


def mix_collate_fn(batch):
    graph_data, query_embedding, bert_score = list(zip(*batch))
    graph_data = np.array(graph_data).flatten()
    graph_data = [dgl.add_self_loop(i) for i in graph_data]
    graph_data = dgl.batch(graph_data)

    query_embedding = [torch.vstack(q) for q in query_embedding]
    bert_score = [torch.from_numpy(bs) for bs in bert_score]

    return graph_data, query_embedding, bert_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--drop", type=float, default=-1)
    parser.add_argument("--coe", type=float, default=-1)
    opt = parser.parse_args()
    
    DATASET = opt.dataset
    SEED = opt.seed
    DROPOUT = opt.drop
    COE = opt.coe
    BATCH_SIZE = opt.batch_size
    MAX_EPOCH = opt.max_epoch
    LR = opt.lr
    IN_DIM = opt.in_dim
    HIDDEN_DIM = opt.hidden_dim
    NUM_LAYER = opt.num_layer
    N_HEAD = opt.n_head

    hyper_configuration = {
        "qmsum": {"dropout": 0.2, "coe": 0.9},
        "wcep": {"dropout": 0.1, "coe": 0.7},
        "booksum": {"dropout": 0.2, "coe": 0.2},
        "govreport": {"dropout": 0.5, "coe": 0.7},
        "squality": {"dropout": 0.1, "coe": 0.4},
    }

    DROPOUT = hyper_configuration[DATASET]["dropout"] if DROPOUT == -1 else DROPOUT
    COE = hyper_configuration[DATASET]["coe"] if COE == -1 else COE

    set_seed(int(SEED))
    DEVICE = get_device(int(opt.cuda))

    gs_list, _ = dgl.load_graphs("./training_data/{}_gs.dgl".format(DATASET))
    query_embedding_list = read_from_pkl(output_file="./training_data/{}_qe.pkl".format(DATASET))
    bert_score_list = read_from_pkl(output_file="./training_data/{}_bs.pkl".format(DATASET))

    train_dataset = GraphDataloader(query_embedding_list=query_embedding_list, gs_list=gs_list,
                                    bert_score_list=bert_score_list)
    train_dataloader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                       collate_fn=mix_collate_fn, num_workers=0, pin_memory=True)
    train_gor(train_dataloader=train_dataloader)