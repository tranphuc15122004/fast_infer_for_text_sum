import argparse
import networkx as nx
import community as community_louvain
from collections import defaultdict
import os
import glob
import torch
import dgl
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple  
from utils import get_llm_response_via_api, read_from_pkl, write_to_pkl, get_device
from retrieval import get_dense_embedding, get_dense_retriever


def detect_communities(graph: nx.Graph) -> (dict, dict):
    """
    Detect communities in the graph using the Louvain algorithm.
    Returns two dictionaries:
    1. partition: {node: community_id}
    2. communities: {community_id: [node1, node2, ...]}
    """
    print("--- Step 1: Start detecting knowledge communities ---")
    partition = community_louvain.best_partition(graph)
    communities = defaultdict(list)
    for node, community_id in partition.items():
        communities[community_id].append(node)
    
    print(f"Successfully detected {len(communities)} knowledge communities.")
    return partition, communities

def summarize_community(node_texts: list, llm_model: str) -> str:
    """
    Summarize all text content within a community using LLM.
    """
    full_text = "\n\n---\n\n".join(node_texts)
    
    prompt = f"""
    As a top-tier Knowledge Architect, your mission is to read and comprehend the following collection of related text fragments.
    Synthesize these scattered pieces of information into a highly condensed, overarching core theme or summary.
    This summary will serve as the "title" or "central idea" for this knowledge cluster and must be both concise and information-dense.

    [Text Fragments to be Summarized]:
    ---
    {full_text}
    ---

    [Core Theme Summary]:
    """
    
    try:
        summary = get_llm_response_via_api(prompt, LLM_MODEL=llm_model)
        if summary and isinstance(summary, str):
            return summary.strip()
        else:
            print(f"Warning: Invalid API response, using fallback summary")
            return f"Knowledge cluster with {len(node_texts)} related text fragments"
    except AttributeError as e:
        print(f"Warning: API response format error: {e}")
        return f"Knowledge cluster with {len(node_texts)} related text fragments"
    except Exception as e:
        print(f"Warning: API call failed: {e}")
        return f"Knowledge cluster with {len(node_texts)} related text fragments"

def create_hierarchical_graph_and_data(
    nx_graph: nx.Graph, 
    dgl_graph: dgl.DGLGraph,
    training_data: list,
    partition: dict,
    communities: dict, 
    llm_model: str,
    device: torch.device,
    retriever_name: str,
    ctx_tokenizer,
    ctx_encoder
) -> (nx.Graph, dgl.DGLGraph, list):
    """
    Create summary nodes for each community and synchronize updates to networkx, dgl graphs, and training_data.
    """
    print("\n--- Step 2: Generate summary nodes and build hierarchy (Sync DGL and PKL) ---")
    
    ctx_encoder.to(device)

    node_to_dgl_id = {node: i for i, node in enumerate(nx_graph.nodes())}

    for community_id, community_nodes in communities.items():
        if len(community_nodes) <= 1:
            continue
        
        node_contents = [nx_graph.nodes[node].get('content', node) for node in community_nodes]
        summary_text = summarize_community(node_contents, llm_model)
        summary_node_id_str = f"SUMMARY_NODE_{community_id}"
        print(f"  > Summary for community {community_id}: '{summary_text[:80]}...'")
        
        summary_embedding = get_dense_embedding(
            [summary_text], retriever=retriever_name, tokenizer=ctx_tokenizer, model=ctx_encoder
        )[0].cpu().unsqueeze(0)

        nx_graph.add_node(summary_node_id_str, content=summary_text, node_type='summary_node')
        dgl_graph.add_nodes(1, {'feat': summary_embedding})
        new_dgl_node_idx = dgl_graph.number_of_nodes() - 1
        node_to_dgl_id[summary_node_id_str] = new_dgl_node_idx

        for node in community_nodes:
            nx_graph.add_edge(summary_node_id_str, node, relationship='summarizes')
            dgl_graph.add_edges(node_to_dgl_id[summary_node_id_str], node_to_dgl_id[node])
            
    augmented_training_data = []
    node_index_to_content = {i: node for i, node in enumerate(nx_graph.nodes()) if 'node_type' not in nx_graph.nodes[node]}

    for sample in training_data:
        new_sample = sample.copy()
        
        response_node_idx = sample['response'][0]
        if response_node_idx in node_index_to_content:
            response_node_content = node_index_to_content[response_node_idx]
            
            if response_node_content in partition:
                community_id = partition[response_node_content]
                summary_node_id_str = f"SUMMARY_NODE_{community_id}"
                
                if summary_node_id_str in node_to_dgl_id:
                    new_sample['summary_response_id'] = node_to_dgl_id[summary_node_id_str]
        
        augmented_training_data.append(new_sample)

    print("\nHierarchical graph and augmented training data construction completed.")
    return nx_graph, dgl_graph, augmented_training_data

def detect_communities_diffusion(
    graph: nx.Graph,
    method: str = "ppr",
    alpha: float = 0.12,
    max_iter: int = 150,
    tol: float = 1e-7,
    num_seeds: int = None,
    adaptive_seeds: bool = True
) -> (dict, dict):
    """
    Diffusion/Random Walk based community detection method (Enhanced)
    
    Args:
        graph: Input graph
        method: "ppr" (PersonalizedPageRank) or "heat_kernel" 
        alpha: Restart probability in PPR, or diffusion strength in heat kernel
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        num_seeds: Number of seed nodes (automatically selected if None)
        adaptive_seeds: Whether to use adaptive seed selection strategy
    """
    print(f"--- Step 1: Start diffusion-based knowledge community detection [{method}] (Enhanced) ---")
    
    nodes = list(graph.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    if n == 0:
        return {}, {}
    
    # Adaptive seed selection strategy
    if adaptive_seeds and num_seeds is None:
        # Select number of seeds based on graph connectivity
        avg_degree = sum(dict(graph.degree()).values()) / n
        if avg_degree > 10:  # Highly connected graph
            num_seeds = max(3, min(8, int(np.log(n))))
        else:  # Sparse graph
            num_seeds = max(2, min(6, int(np.sqrt(n) * 0.8)))
        print(f"Adaptive seed count selection: {num_seeds} (Average degree: {avg_degree:.2f})")
    elif num_seeds is None:
        num_seeds = max(2, min(10, int(np.sqrt(n))))
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(graph, nodelist=nodes).astype(float)
    
    if method == "ppr":
        # Enhanced PPR calculation
        influence_matrix = compute_enhanced_ppr_matrix(A, alpha, max_iter, tol)
    elif method == "heat_kernel":
        # Heat Kernel diffusion
        influence_matrix = compute_heat_kernel_matrix(A, alpha)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Intelligent seed selection: Based on PPR influence only
    print("Seed selection based on PPR influence...")
    
    # Calculate total PPR influence for each node
    ppr_influence = np.sum(influence_matrix, axis=1)
    
    # Select nodes with highest PPR influence as seeds
    seed_indices = np.argsort(ppr_influence)[-num_seeds:]
    
    # Clustering based on influence domain, but with overlap handling
    seed_influences = influence_matrix[seed_indices, :]
    
    partition = {}
    overlap_threshold = 0.1  # Handle boundary nodes
    
    for i, node in enumerate(nodes):
        influences = seed_influences[:, i]
        max_influence = np.max(influences)
        dominant_seed = np.argmax(influences)
        
        # Handle boundary nodes: if influences from multiple seeds are close, assign to the strongest one
        if max_influence < overlap_threshold:
            # Low influence nodes assigned to the nearest seed
            try:
                distances = []
                for seed_idx in seed_indices:
                    try:
                        dist = nx.shortest_path_length(graph, nodes[seed_idx], node, cutoff=3)
                        distances.append(dist)
                    except nx.NetworkXNoPath:
                        distances.append(float('inf'))
                
                if distances and min(distances) != float('inf'):
                    dominant_seed = distances.index(min(distances))
            except:
                # If distance calculation fails, keep original assignment
                pass
        
        partition[node] = int(dominant_seed)
    
    # Build community dictionary
    communities = defaultdict(list)
    for node, community_id in partition.items():
        communities[community_id].append(node)
    
    print(f"Detected {len(communities)} PPR communities")
    print(f"Seed nodes: {[nodes[i] for i in seed_indices]}")
    print(f"Average community size: {np.mean([len(comm) for comm in communities.values()]):.1f}")
    
    return partition, communities

def compute_ppr_matrix(A, alpha=0.15, max_iter=100, tol=1e-6):
    """
    Calculate Personalized PageRank matrix
    Each row represents the PPR distribution with that node as seed
    """
    n = A.shape[0]
    
    # Transition matrix (column stochastic)
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # Avoid division by zero
    D_inv = np.diag(1.0 / degrees)
    P = A.T @ D_inv  # Transition matrix
    
    # Calculate PPR for each node
    ppr_matrix = np.zeros((n, n))
    
    for seed in range(n):
        # Initialization: only seed node has probability mass
        r = np.zeros(n)
        r[seed] = 1.0
        
        for _ in range(max_iter):
            r_new = alpha * np.zeros(n)
            r_new[seed] = alpha  
            r_new += (1 - alpha) * (P @ r)
            
            if np.linalg.norm(r_new - r) < tol:
                break
            r = r_new
        
        ppr_matrix[seed, :] = r
    
    return ppr_matrix

def compute_enhanced_ppr_matrix(A, alpha=0.12, max_iter=150, tol=1e-7):
    """
    Enhanced PPR calculation with preprocessing and convergence optimization
    """
    n = A.shape[0]
    
    # Handle isolated nodes and numerical stability
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # Avoid division by zero
    
    # Add small random perturbation to avoid convergence to local optima
    A_smoothed = A.copy().astype(float)
    if hasattr(A, 'toarray'):
        A_smoothed = A_smoothed.toarray()
    A_smoothed += 1e-8 * np.random.rand(n, n)
    
    degrees_smoothed = np.array(A_smoothed.sum(axis=1)).flatten()
    degrees_smoothed[degrees_smoothed == 0] = 1
    D_inv = np.diag(1.0 / degrees_smoothed)
    P = A_smoothed.T @ D_inv
    
    ppr_matrix = np.zeros((n, n))
    
    for seed in range(n):
        r = np.zeros(n)
        r[seed] = 1.0
        
        # Use momentum to accelerate convergence
        r_prev = r.copy()
        momentum = 0.9
        converged = False
        
        for iter_count in range(max_iter):
            r_new = alpha * np.zeros(n)
            r_new[seed] = alpha
            r_new += (1 - alpha) * (P @ r)
            
            # Add momentum term
            if iter_count > 0:
                r_new = r_new + momentum * (r - r_prev)
            
            # Numerical stability: normalization
            if np.sum(r_new) > 0:
                r_new = r_new / np.sum(r_new)
            
            if np.linalg.norm(r_new - r) < tol:
                converged = True
                break
                
            r_prev = r.copy()
            r = r_new
        
        if not converged and seed % 100 == 0:  # Print warning only for a few seeds
            print(f"Warning: PPR for seed {seed} did not converge after {max_iter} iterations")
        
        ppr_matrix[seed, :] = r
    
    # Post-processing: ensure row sums are 1 (numerical stability)
    row_sums = np.sum(ppr_matrix, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    ppr_matrix = ppr_matrix / row_sums
    
    return ppr_matrix

def compute_heat_kernel_matrix(A, t=1.0):
    """
    Calculate Heat Kernel diffusion matrix
    H = exp(-t * L), where L is the Laplacian matrix
    """
    # Laplacian matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D = np.diag(degrees)
    L = D - A.toarray() if hasattr(A, 'toarray') else D - A
    
    # Normalized Laplacian matrix (avoid numerical issues)
    degrees[degrees == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Heat kernel: exp(-t * L_norm)
    eigenvals, eigenvecs = np.linalg.eigh(L_norm)
    heat_kernel = eigenvecs @ np.diag(np.exp(-t * eigenvals)) @ eigenvecs.T
    
    # Transform back to original space
    D_sqrt = np.diag(np.sqrt(degrees))
    heat_kernel = D_sqrt @ heat_kernel @ D_sqrt
    
    return heat_kernel

def create_diffusion_based_summary(
    graph: nx.Graph,
    partition: dict,  
    communities: dict,  
    influence_matrix: np.ndarray,
    nodes: list  
) -> dict: 
    """
    Create smarter summaries based on diffusion influence
    Consider not only community content but also influence propagation paths
    """
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    summaries = {}
    
    for community_id, community_nodes in communities.items():
        # Find "core influence nodes" of this community
        community_indices = [node_to_idx[node] for node in community_nodes]
        
        # Calculate average influence of nodes within the community
        community_influence = influence_matrix[community_indices, :]
        avg_influence = np.mean(community_influence, axis=0)
        
        # Find external areas most influenced by this community
        external_influenced = []
        for i, influence_score in enumerate(avg_influence):
            if nodes[i] not in community_nodes and influence_score > np.percentile(avg_influence, 75):
                external_influenced.append(nodes[i])
        
        # Build influence domain description
        summaries[community_id] = {
            "core_nodes": community_nodes,
            "influence_extent": external_influenced[:5],  # Limit quantity
            "influence_strength": float(np.sum(avg_influence)),
            "coverage_ratio": len(external_influenced) / len(nodes)
        }
    
    return summaries

def evaluate_community_quality(graph: nx.Graph, communities: dict) -> dict:
    """
    Evaluate community quality metrics
    """
    metrics = {}
    
    # Calculate modularity (Modularity)
    try:
        partition = {}
        for comm_id, nodes in communities.items():
            for node in nodes:
                partition[node] = comm_id
        
        modularity = nx.algorithms.community.modularity(graph, communities.values())
        metrics['modularity'] = modularity
    except:
        metrics['modularity'] = 0.0
    
    # Community size distribution
    sizes = [len(comm) for comm in communities.values()]
    metrics['avg_community_size'] = np.mean(sizes)
    metrics['std_community_size'] = np.std(sizes)
    metrics['max_community_size'] = np.max(sizes)
    metrics['min_community_size'] = np.min(sizes)
    
    # Intra-community connectivity
    internal_edges = 0
    total_edges = 0
    
    for comm_nodes in communities.values():
        subgraph = graph.subgraph(comm_nodes)
        internal_edges += subgraph.number_of_edges()
    
    total_edges = graph.number_of_edges()
    metrics['internal_edge_ratio'] = internal_edges / max(total_edges, 1)
    
    print(f"Community quality evaluation:")
    print(f"  Modularity: {metrics['modularity']:.3f}")
    print(f"  Average community size: {metrics['avg_community_size']:.1f}")
    print(f"  Internal edge ratio: {metrics['internal_edge_ratio']:.3f}")
    
    return metrics

def adaptive_parameter_tuning(graph: nx.Graph, method: str = "ppr") -> dict:
    """
    Automatically adjust PPR parameters based on graph characteristics
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    
    if n == 0:
        return {"alpha": 0.15, "max_iter": 100, "tol": 1e-6}
    
    # Basic graph statistics
    avg_degree = 2 * m / n if n > 0 else 0
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    
    # Connected component analysis
    num_components = nx.number_connected_components(graph)
    largest_cc_size = len(max(nx.connected_components(graph), key=len)) if num_components > 0 else 0
    
    # Adaptive parameters
    params = {}
    
    if method == "ppr":
        # Dense graphs use lower alpha (more diffusion)
        if density > 0.1:
            params["alpha"] = 0.08
        elif density > 0.05:
            params["alpha"] = 0.12
        else:
            params["alpha"] = 0.15
        
        # Large graphs require more iterations
        if n > 1000:
            params["max_iter"] = 200
            params["tol"] = 1e-8
        elif n > 500:
            params["max_iter"] = 150
            params["tol"] = 1e-7
        else:
            params["max_iter"] = 100
            params["tol"] = 1e-6
    
    print(f"Adaptive parameter tuning:")
    print(f"  Graph size: {n} nodes, {m} edges")
    print(f"  Average degree: {avg_degree:.2f}, Density: {density:.4f}")
    print(f"  Connected components: {num_components}, Largest component: {largest_cc_size}")
    print(f"  Adjusted parameters: {params}")
    
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch add knowledge synthesis and summary layer to GoR graphs (Sync DGL).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--input-dir", type=str, default="./graph", help="Directory of input graph files.")
    parser.add_argument("--output-dir", type=str, default="./graph_hierarchical", help="Directory of output hierarchical graphs.")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1-mini-2025-04-14", help="Summary LLM model.")
    parser.add_argument("--retriever", type=str, default="contriever", help="Retriever model name for generating embeddings.")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device.")
    
    args = parser.parse_args()
    
    DEVICE = get_device(int(args.cuda))
    
    # Initialize retriever/embedding components
    _, CTX_TOKENIZER, _, CTX_ENCODER = get_dense_retriever(retriever=args.retriever)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Search for training and test graphs
    train_pattern = os.path.join(args.input_dir, f"{args.dataset}_graph_*.graphml")
    test_pattern = os.path.join(args.input_dir, f"{args.dataset}_test_graph_*.graphml")
    
    train_files = glob.glob(train_pattern)
    test_files = glob.glob(test_pattern)
    graphml_files = train_files + test_files
    
    if not graphml_files:
        print(f"No matching graph files found in directory '{args.input_dir}'.")
        print(f"Search patterns: {train_pattern} and {test_pattern}")
    else:
        print(f"Found {len(train_files)} training graph files and {len(test_files)} test graph files, total {len(graphml_files)} files, preparing to process...")

    for graphml_path in graphml_files:
        print(f"\n=================================================")
        print(f"Processing file series: {os.path.basename(graphml_path)}")
        print(f"=================================================")

        dgl_path = graphml_path.replace('.graphml', '.dgl')
        
        # Generate corresponding pkl file path and output file path based on file type (training/test)
        if '_test_graph_' in os.path.basename(graphml_path):
            # Test file processing - no pkl file needed for test files
            pkl_path = None  # No corresponding training data file for test files
            output_graphml_path = os.path.join(args.output_dir, os.path.basename(graphml_path).replace('_test_graph_', '_test_hierarchical_graph_'))
            output_dgl_path = os.path.join(args.output_dir, os.path.basename(dgl_path).replace('_test_graph_', '_test_hierarchical_graph_'))
            output_pkl_path = None  # No pkl output for test files
        else:
            # Training file processing
            pkl_path = graphml_path.replace('_graph_', '_training_data_').replace('.graphml', '.pkl')
            output_graphml_path = os.path.join(args.output_dir, os.path.basename(graphml_path).replace('_graph_', '_hierarchical_graph_'))
            output_dgl_path = os.path.join(args.output_dir, os.path.basename(dgl_path).replace('_graph_', '_hierarchical_graph_'))
            output_pkl_path = os.path.join(args.output_dir, os.path.basename(pkl_path).replace('_training_data_', '_hierarchical_training_data_'))
   
        # --- Check if output files already exist based on file type ---
        if '_test_graph_' in os.path.basename(graphml_path):
            # Test files only check graphml and dgl
            if os.path.exists(output_graphml_path) and os.path.exists(output_dgl_path):
                print(f"  > Found existing test file output. Skipping this series.")
                continue
        else:
            # Training files check all three files
            if os.path.exists(output_graphml_path) and os.path.exists(output_dgl_path) and os.path.exists(output_pkl_path):
                print(f"  > Found existing training file output. Skipping this series.")
                continue

        # --- Check if input files exist based on file type ---
        if '_test_graph_' in os.path.basename(graphml_path):
            # Test files only need dgl file to exist
            if not os.path.exists(dgl_path):
                print(f"  > Warning: Pair .dgl file not found, skipping this test file.")
                continue
        else:
            # Training files need both dgl and pkl to exist
            if not (os.path.exists(dgl_path) and os.path.exists(pkl_path)):
                print(f"  > Warning: Pair .dgl or .pkl file not found, skipping this training file.")
                continue
        
        # --- GraphML file parsing exception handling ---
        try:
            nx_graph = nx.read_graphml(graphml_path)
        except Exception as e:
            print(f"  > Error: Unable to parse GraphML file '{graphml_path}': {e}")
            print(f"  > Skipping this corrupted file, continue processing the next one.")
            continue
        
        try:
            dgl_graphs, _ = dgl.load_graphs(dgl_path)
            dgl_graph = dgl_graphs[0]
            
            # Load training data based on file type
            if '_test_graph_' in os.path.basename(graphml_path):
                # Test files do not need training data
                training_data = []  # Empty list
            else:
                # Training files need to load training data
                training_data = read_from_pkl(pkl_path)
        except Exception as e:
            print(f"  > Error: Unable to load DGL or PKL files: {e}")
            print(f"  > Skipping this file series, continue processing the next one.")
            continue
        
        # partition, communities = detect_communities(nx_graph)
        adaptive_params = adaptive_parameter_tuning(nx_graph, method="ppr")

        # Use diffusion-based community detection with adaptive parameters
        partition, communities = detect_communities_diffusion(
            nx_graph, 
            method="ppr",                              # or "heat_kernel"
            alpha=adaptive_params.get("alpha", 0.12), 
            max_iter=adaptive_params.get("max_iter", 150),
            tol=adaptive_params.get("tol", 1e-7),
            num_seeds=None,                            
            adaptive_seeds=True                        
        )
        
        # evaluate_community_quality
        quality_metrics = evaluate_community_quality(nx_graph, communities)
        
        hierarchical_nx_graph, hierarchical_dgl_graph, augmented_data = create_hierarchical_graph_and_data(
            nx_graph, dgl_graph, training_data, partition, communities, args.llm_model, DEVICE,
            args.retriever, CTX_TOKENIZER, CTX_ENCODER
        )
        
        nx.write_graphml(hierarchical_nx_graph, output_graphml_path)
        dgl.save_graphs(output_dgl_path, [hierarchical_dgl_graph])
        
        if '_test_graph_' in os.path.basename(graphml_path):
            print(f"\nSuccessfully saved the hierarchical graph of the test graph to '{args.output_dir}' directory.")
        else:
            write_to_pkl(augmented_data, output_pkl_path)
            print(f"\nSuccessfully saved the hierarchical graph triples of the training graph to '{args.output_dir}' directory.")
    
    print("\nAll files processed!")
