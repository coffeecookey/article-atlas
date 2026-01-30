import json
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import networkx as nx
from networkx.algorithms import community

Entity = Dict[str, Any]
Relationship = Dict[str, Any]
GraphData = Dict[str, Any]

#import entity, relationship and graphData pydantic models

def build_graph(
    entities: List[Entity],
    relationships: List[Relationship]
):
    print(f"\n{'='*70}")
    print(f"BUILDING KNOWLEDGE GRAPH")
    print(f"{'='*70}")
    print(f"Entities: {len(entities)}")
    print(f"Relationships: {len(relationships)}")
    
    G = nx.DiGraph()
    
    node_count = 0
    for entity in entities:
        entity_id = entity.get("id")
        if not entity_id:
            print(f"Warning: Entity missing ID, skipping: {entity}")
            continue
        
        G.add_node(
            entity_id,
            name=entity.get("name", ""),
            type=entity.get("type", "UNKNOWN"),
            mentions=entity.get("mentions", 1),
            chunk_ids=entity.get("chunk_ids", []),
            properties=entity.get("properties", {}),
            canonical_name=entity.get("canonical_name", entity.get("name", "")),
            aliases=entity.get("aliases", [])
        )
        node_count += 1
    
    print(f"Added {node_count} nodes")
    
    edge_count = 0
    skipped_edges = 0
    
    for rel in relationships:
        source = rel.get("source")
        target = rel.get("target")
        rel_type = rel.get("type", "RELATED_TO")
        
        if not source or not target:
            skipped_edges += 1
            continue
        
        if source not in G or target not in G:
            skipped_edges += 1
            continue
        
        G.add_edge(
            source,
            target,
            type=rel_type,
            properties=rel.get("properties", {}),
            edge_id=f"{source}__{rel_type}__{target}"
        )
        edge_count += 1
    
    print(f"Added {edge_count} edges")
    if skipped_edges > 0:
        print(f"Skipped {skipped_edges} edges (missing source/target nodes)")
    
    metrics = calculate_basic_metrics(G)
    
    print(f"\n Graph Statistics:")
    print(f"  Nodes: {metrics['node_count']}")
    print(f"  Edges: {metrics['edge_count']}")
    print(f"  Density: {metrics['density']:.4f}")
    print(f"  Connected: {metrics['is_connected']}")
    if metrics['is_connected']:
        print(f"  Diameter: {metrics.get('diameter', 'N/A')}")
    else:
        print(f"  Connected components: {metrics['connected_components']}")
    
    return {
        "graph": G,
        "entities": entities,
        "relationships": relationships,
        "metrics": metrics
    }


def calculate_basic_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """Calculate basic graph metrics"""
    metrics = {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "density": nx.density(G),
        "is_directed": G.is_directed()
    }
    
    if G.is_directed():
        metrics["is_connected"] = nx.is_weakly_connected(G)
        metrics["connected_components"] = nx.number_weakly_connected_components(G)
    else:
        metrics["is_connected"] = nx.is_connected(G)
        metrics["connected_components"] = nx.number_connected_components(G)
    
    if metrics["is_connected"] and metrics["node_count"] > 1:
        try:
            if G.is_directed():
                metrics["diameter"] = nx.diameter(G.to_undirected())
            else:
                metrics["diameter"] = nx.diameter(G)
        except:
            metrics["diameter"] = None
    
    return metrics

def enrich_graph(graph_data: GraphData):
    """
    Enrich graph with computed properties and analysis
    """
    print(f"\n{'='*70}")
    print(f"ENRICHING KNOWLEDGE GRAPH")
    print(f"{'='*70}")
    
    G = graph_data["graph"]
    
    if G.number_of_nodes() == 0:
        print("Warning: Empty graph, skipping enrichment")
        return graph_data
    
    print("\n Calculating centrality measures...")
    centralities = calculate_centrality_measures(G)
    
    for node_id in G.nodes():
        G.nodes[node_id]["degree_centrality"] = centralities["degree"].get(node_id, 0)
        G.nodes[node_id]["betweenness_centrality"] = centralities["betweenness"].get(node_id, 0)
        G.nodes[node_id]["pagerank"] = centralities["pagerank"].get(node_id, 0)
        
        if "eigenvector" in centralities:
            G.nodes[node_id]["eigenvector_centrality"] = centralities["eigenvector"].get(node_id, 0)
    
    print("🔍 Detecting communities...")
    communities = detect_communities(G)
    
    for i, community_nodes in enumerate(communities):
        for node_id in community_nodes:
            if node_id in G.nodes():
                G.nodes[node_id]["community"] = i
    
    print(f"Found {len(communities)} communities")
    
    print("🛤️  Computing key paths...")
    key_paths = find_key_paths(G, centralities, max_paths=10)
    
    if G.is_directed() and G.number_of_nodes() > 1:
        print("⭐ Calculating hub and authority scores...")
        try:
            hubs, authorities = nx.hits(G, max_iter=100)
            for node_id in G.nodes():
                G.nodes[node_id]["hub_score"] = hubs.get(node_id, 0)
                G.nodes[node_id]["authority_score"] = authorities.get(node_id, 0)
        except:
            print("Warning: Could not calculate HITS scores")
    
    enriched_metrics = {
        **graph_data["metrics"],
        "centrality_stats": get_centrality_statistics(centralities),
        "communities": len(communities),
        "key_paths": key_paths
    }
    
    print(f"\n Enrichment complete!")
    print(f"  Top 5 nodes by PageRank:")
    top_nodes = sorted(
        centralities["pagerank"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for node_id, score in top_nodes:
        node_name = G.nodes[node_id].get("name", node_id)
        print(f"    {node_name}: {score:.4f}")
    
    return {
        **graph_data,
        "metrics": enriched_metrics,
        "communities": communities
    }


def calculate_centrality_measures(G: nx.DiGraph) -> Dict[str, Dict]:
    """Calculate various centrality measures"""
    centralities = {}
    
    if G.is_directed():
        centralities["degree"] = nx.in_degree_centrality(G)
        centralities["out_degree"] = nx.out_degree_centrality(G)
    else:
        centralities["degree"] = nx.degree_centrality(G)
    
    try:
        centralities["betweenness"] = nx.betweenness_centrality(G)
    except:
        centralities["betweenness"] = {node: 0 for node in G.nodes()}
    
    try:
        centralities["pagerank"] = nx.pagerank(G, max_iter=100)
    except:
        centralities["pagerank"] = {node: 1.0 / G.number_of_nodes() for node in G.nodes()}
    
    try:
        centralities["eigenvector"] = nx.eigenvector_centrality(G, max_iter=100)
    except:
        print("Warning: Eigenvector centrality did not converge")
    
    return centralities


def detect_communities(G: nx.DiGraph) -> List[Set[str]]:
    if G.number_of_nodes() < 2:
        return [set(G.nodes())]
    
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    try:
        communities_generator = community.greedy_modularity_communities(G_undirected)
        communities = [set(c) for c in communities_generator]
        return communities
    except:
        return [set([node]) for node in G.nodes()]


def find_key_paths(
    G: nx.DiGraph,
    centralities: Dict[str, Dict],
    max_paths: int = 10
) -> List[Dict[str, Any]]:
    if G.number_of_nodes() < 2:
        return []
    
    top_nodes = sorted(
        centralities["pagerank"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:min(10, G.number_of_nodes())]
    
    top_node_ids = [node_id for node_id, _ in top_nodes]
    
    paths = []
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    for i, source in enumerate(top_node_ids):
        for target in top_node_ids[i+1:]:
            if source != target:
                try:
                    path = nx.shortest_path(G_undirected, source, target)
                    paths.append({
                        "source": source,
                        "target": target,
                        "length": len(path) - 1,
                        "path": path,
                        "source_name": G.nodes[source].get("name", source),
                        "target_name": G.nodes[target].get("name", target)
                    })
                except nx.NetworkXNoPath:
                    continue
            
            if len(paths) >= max_paths:
                break
        if len(paths) >= max_paths:
            break
    
    paths.sort(key=lambda x: x["length"])
    
    return paths[:max_paths]


def get_centrality_statistics(centralities: Dict[str, Dict]) -> Dict[str, Any]:
    """Get statistics about centrality measures"""
    stats = {}
    
    for measure_name, scores in centralities.items():
        if not scores:
            continue
        
        values = list(scores.values())
        stats[measure_name] = {
            "mean": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "top_5": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    return stats


def export_to_json(graph_data: GraphData) -> Dict[str, Any]:
    """
    Export graph to JSON format for frontend consumption
    """
    print(f"\n{'='*70}")
    print(f"EXPORTING GRAPH TO JSON")
    print(f"{'='*70}")
    
    G = graph_data["graph"]
    
    nodes = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        
        node = {
            "id": node_id,
            "label": node_data.get("name", node_id),
            "type": node_data.get("type", "UNKNOWN"),
            "properties": {
                "mentions": node_data.get("mentions", 1),
                "canonical_name": node_data.get("canonical_name", ""),
                "aliases": node_data.get("aliases", []),
                "chunk_ids": node_data.get("chunk_ids", []),
                **node_data.get("properties", {})
            },
            "metrics": {
                "degree_centrality": node_data.get("degree_centrality", 0),
                "betweenness_centrality": node_data.get("betweenness_centrality", 0),
                "pagerank": node_data.get("pagerank", 0),
                "eigenvector_centrality": node_data.get("eigenvector_centrality", 0),
                "hub_score": node_data.get("hub_score", 0),
                "authority_score": node_data.get("authority_score", 0),
            },
            "community": node_data.get("community", -1)
        }
        
        nodes.append(node)
    
    edges = []
    for source, target, edge_data in G.edges(data=True):
        edge = {
            "id": edge_data.get("edge_id", f"{source}__{target}"),
            "source": source,
            "target": target,
            "label": edge_data.get("type", "RELATED_TO"),
            "type": edge_data.get("type", "RELATED_TO"),
            "properties": edge_data.get("properties", {})
        }
        
        edges.append(edge)
    
    metadata = {
        "metrics": sanitize_for_json(graph_data.get("metrics", {})),
        "communities": [
            {
                "id": i,
                "size": len(comm),
                "nodes": list(comm)
            }
            for i, comm in enumerate(graph_data.get("communities", []))
        ],
        "node_count": len(nodes),
        "edge_count": len(edges)
    }
    
    result = {
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata
    }
    
    print(f"Exported {len(nodes)} nodes and {len(edges)} edges")
    
    return result


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return sanitize_for_json(obj.__dict__)
    else:
        return obj


def export_to_file(graph_data: GraphData, filepath: str) -> None:
    json_data = export_to_json(graph_data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved graph to: {filepath}")


def find_node_by_name(graph_data: GraphData, name: str) -> Optional[str]:
    G = graph_data["graph"]
    name_lower = name.lower()
    
    for node_id in G.nodes():
        node_name = G.nodes[node_id].get("name", "").lower()
        if node_name == name_lower:
            return node_id
        
        aliases = G.nodes[node_id].get("aliases", [])
        if any(alias.lower() == name_lower for alias in aliases):
            return node_id
    
    return None


def get_node_neighbors(
    graph_data: GraphData,
    node_id: str,
    depth: int = 1
) -> Dict[str, Any]:
    G = graph_data["graph"]
    
    if node_id not in G:
        return {"nodes": [], "edges": []}
    
    neighbors = {node_id}
    current_layer = {node_id}
    
    for _ in range(depth):
        next_layer = set()
        for node in current_layer:
            next_layer.update(G.predecessors(node))
            next_layer.update(G.successors(node))
        neighbors.update(next_layer)
        current_layer = next_layer
    
    subgraph = G.subgraph(neighbors)
    
    nodes = []
    for n in subgraph.nodes():
        node_data = G.nodes[n]
        nodes.append({
            "id": n,
            "label": node_data.get("name", n),
            "type": node_data.get("type", "UNKNOWN"),
            "properties": node_data.get("properties", {})
        })
    
    edges = []
    for source, target in subgraph.edges():
        edge_data = G.edges[source, target]
        edges.append({
            "source": source,
            "target": target,
            "label": edge_data.get("type", "RELATED_TO"),
            "properties": edge_data.get("properties", {})
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "center_node": node_id
    }

def get_graph_summary(graph_data: GraphData) -> Dict[str, Any]:
    G = graph_data["graph"]
    metrics = graph_data.get("metrics", {})
    
    entity_types = defaultdict(int)
    for node in G.nodes():
        node_type = G.nodes[node].get("type", "UNKNOWN")
        entity_types[node_type] += 1
    
    relationship_types = defaultdict(int)
    for _, _, data in G.edges(data=True):
        rel_type = data.get("type", "RELATED_TO")
        relationship_types[rel_type] += 1
    
    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "density": metrics.get("density", 0),
        "communities": metrics.get("communities", 0),
        "entity_types": dict(entity_types),
        "relationship_types": dict(relationship_types),
        "is_connected": metrics.get("is_connected", False)
    }
