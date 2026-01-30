import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import networkx as nx
from networkx.readwrite import json_graph

DEFAULT_STORAGE_DIR = "./data/graphs"
Path(DEFAULT_STORAGE_DIR).mkdir(parents=True, exist_ok=True)

GraphData = Dict[str, Any]

def save_to_json(graph_data: GraphData, filepath: str, include_metadata: bool = True) -> bool:
    try:
        filepath = _normalize_filepath(filepath, extension=".json")

        print("Saving graph to JSON")
        print(f"File: {filepath}")

        G = graph_data.get("graph")
        if G is None:
            raise ValueError("GraphData must contain 'graph' key with NetworkX graph")

        graph_json = json_graph.node_link_data(G)

        output = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "graph_type": "directed" if G.is_directed() else "undirected"
            },
            "graph": graph_json,
            "entities": graph_data.get("entities", []),
            "relationships": graph_data.get("relationships", [])
        }

        if include_metadata:
            output["metrics"] = graph_data.get("metrics", {})
            if "communities" in graph_data:
                output["communities"] = [list(c) for c in graph_data.get("communities", [])]

        output = _sanitize_for_json(output)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        file_size = os.path.getsize(filepath)
        print("Saved successfully")
        print(f"File size: {file_size}")
        print(f"Nodes: {output['metadata']['node_count']}")
        print(f"Edges: {output['metadata']['edge_count']}")

        return True

    except Exception as e:
        print(f"Failed to save graph: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_from_json(filepath: str) -> Optional[GraphData]:
    try:
        filepath = _normalize_filepath(filepath, extension=".json")

        print("Loading graph from JSON")
        print(f"File: {filepath}")

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        G = json_graph.node_link_graph(data["graph"])

        graph_data = {
            "graph": G,
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
            "metrics": data.get("metrics", {})
        }

        if "communities" in data:
            graph_data["communities"] = [set(c) for c in data["communities"]]

        print("Loaded successfully")
        print(f"Nodes: {data['metadata']['node_count']}")
        print(f"Edges: {data['metadata']['edge_count']}")

        return graph_data

    except Exception as e:
        print(f"Failed to load graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_graphml(graph_data: GraphData, filepath: str) -> bool:
    try:
        filepath = _normalize_filepath(filepath, extension=".graphml")

        print("Exporting to GraphML")
        print(f"File: {filepath}")

        G = graph_data.get("graph")
        if G is None:
            raise ValueError("GraphData must contain 'graph' key")

        G_export = G.copy()

        for node in G_export.nodes():
            node_data = G_export.nodes[node]
            for key, value in list(node_data.items()):
                if isinstance(value, (list, dict, set)):
                    node_data[key] = json.dumps(value)

        for source, target in G_export.edges():
            edge_data = G_export.edges[source, target]
            for key, value in list(edge_data.items()):
                if isinstance(value, (list, dict, set)):
                    edge_data[key] = json.dumps(value)

        nx.write_graphml(G_export, filepath)

        file_size = os.path.getsize(filepath)
        print("Exported successfully")
        print(f"File size: {file_size}")

        return True

    except Exception as e:
        print(f"Failed to export to GraphML: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_saved_graphs(directory: str = DEFAULT_STORAGE_DIR) -> List[Dict[str, Any]]:
    graphs = []

    if not os.path.exists(directory):
        return graphs

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)

            try:
                stat = os.stat(filepath)

                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = data.get("metadata", {})

                graphs.append({
                    "name": filename.replace('.json', ''),
                    "filepath": filepath,
                    "size_bytes": stat.st_size,
                    "size_kb": stat.st_size / 1024,
                    "created_at": metadata.get("created_at", "Unknown"),
                    "node_count": metadata.get("node_count", 0),
                    "edge_count": metadata.get("edge_count", 0),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except:
                continue

    graphs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return graphs


def _normalize_filepath(filepath: str, extension: str = ".json") -> str:
    if not os.path.dirname(filepath):
        filepath = os.path.join(DEFAULT_STORAGE_DIR, filepath)

    if not filepath.endswith(extension):
        base = os.path.splitext(filepath)[0]
        filepath = base + extension

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    return filepath


def _sanitize_for_json(obj: Any) -> Any:
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return _sanitize_for_json(obj.__dict__)
    else:
        return str(obj)


def save_graph(graph_data: GraphData, name: str, export_graphml: bool = False) -> bool:
    success = save_to_json(graph_data, name)

    if success and export_graphml:
        save_to_graphml(graph_data, name)

    return success


def load_graph(name: str) -> Optional[GraphData]:
    return load_from_json(name)
