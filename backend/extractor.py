import time
import json
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.config import GOOGLE_API_KEY, LLM_ENABLED
# backend.config seems to make fastapi server work but tests fail

EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting entities and relationships from text to build knowledge graphs.

Extract the following types of entities:
- PERSON: People, historical figures, authors
- ORGANIZATION: Companies, institutions, government bodies
- LOCATION: Cities, countries, regions, places
- CONCEPT: Ideas, theories, topics, technologies
- EVENT: Historical events, occurrences, incidents
- DATE: Specific dates, time periods

Extract relationships between entities:
- WORKS_AT: Person works at Organization
- LOCATED_IN: Entity is located in Location
- FOUNDED: Person/Organization founded Entity
- RELATED_TO: General relationship between Concepts
- PARTICIPATED_IN: Person/Organization participated in Event
- OCCURRED_ON: Event occurred on Date
- PART_OF: Entity is part of another Entity

Return a JSON object with:
{
  "entities": [
    {
      "id": "unique_id",
      "name": "Entity Name",
      "type": "ENTITY_TYPE",
      "properties": {"key": "value"}
    }
  ],
  "relationships": [
    {
      "source": "entity_id_1",
      "target": "entity_id_2",
      "type": "RELATIONSHIP_TYPE",
      "properties": {"key": "value"}
    }
  ]
}

Be precise and only extract entities and relationships that are explicitly mentioned in the text.
"""

def create_extraction_prompt(text: str, schema: Optional[Dict] = None) -> str:
    if schema and "entity_types" in schema and "relationship_types" in schema:
        entity_types = ", ".join(schema["entity_types"])
        rel_types = ", ".join(schema["relationship_types"])

        return f"""Extract entities of types: {entity_types}
Extract relationships of types: {rel_types}

Text:
{text}

Return structured JSON only."""
    else:
        return f"""Extract all relevant entities and relationships from the following text:

{text}

Return structured JSON only."""

_model = None


def get_model():
    global _model
    if not LLM_ENABLED:
        return None

    if _model is None:
        genai.configure(api_key=GOOGLE_API_KEY)
        _model = genai.GenerativeModel("gemini-2.5-flash")

    return _model

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_entities_and_relationships(
    chunk: Dict,
    schema: Optional[Dict] = None
) -> Dict:

    chunk_text = chunk.get("text", "")
    chunk_id = chunk.get("chunk_id", "unknown")

    if not chunk_text:
        return {
            "entities": [],
            "relationships": [],
            "chunk_id": chunk_id
        }

    try:
        model = get_model()
        if model is None:
            return {
                "entities": [],
                "relationships": [],
                "chunk_id": chunk_id,
                "error": "LLM disabled"
            }

        prompt = f"""{EXTRACTION_SYSTEM_PROMPT}

{create_extraction_prompt(chunk_text, schema)}
"""

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        )

        result = json.loads(response.text)

        for entity in result.get("entities", []):
            if "id" not in entity:
                entity["id"] = (
                    f"{entity.get('name', 'unknown')}"
                    .lower()
                    .replace(" ", "_")
                    + f"_{chunk_id}"
                )

        return {
            "entities": result.get("entities", []),
            "relationships": result.get("relationships", []),
            "chunk_id": chunk_id
        }

    except Exception as e:
        print(f"[ERROR] Chunk {chunk_id}: {e}")
        return {
            "entities": [],
            "relationships": [],
            "chunk_id": chunk_id,
            "error": str(e)
        }

def batch_extract(
    chunks: List[Dict],
    schema: Optional[Dict] = None,
    batch_size: int = 5
) -> List[Dict]:

    extractions: List[Dict] = []
    total_chunks = len(chunks)

    print(f"Starting extraction for {total_chunks} chunks...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {(i // batch_size) + 1}")

        for chunk in batch:
            result = extract_entities_and_relationships(chunk, schema)
            extractions.append(result)
            time.sleep(0.5)  # rate limiting

    print("Extraction complete.")
    return extractions

def merge_extractions(extractions: List[Dict]) -> Dict:
    merged_entities: Dict = {}
    merged_relationships: List[Dict] = []

    for extraction in extractions:
        for entity in extraction.get("entities", []):
            key = (entity.get("name", "").lower(), entity.get("type", ""))

            if key not in merged_entities:
                merged_entities[key] = {
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "type": entity.get("type"),
                    "properties": entity.get("properties", {}),
                    "mentions": 1,
                    "chunk_ids": [extraction.get("chunk_id")]
                }
            else:
                merged_entities[key]["mentions"] += 1
                merged_entities[key]["chunk_ids"].append(extraction.get("chunk_id"))

    id_map = {}
    for (name, etype), data in merged_entities.items():
        new_id = f"{name.replace(' ', '_')}_{etype.lower()}"
        id_map[data["id"]] = new_id
        data["id"] = new_id

    seen = set()
    for extraction in extractions:
        for rel in extraction.get("relationships", []):
            src = id_map.get(rel.get("source"), rel.get("source"))
            tgt = id_map.get(rel.get("target"), rel.get("target"))
            rtype = rel.get("type")

            key = (src, tgt, rtype)
            if key not in seen:
                merged_relationships.append({
                    "source": src,
                    "target": tgt,
                    "type": rtype,
                    "properties": rel.get("properties", {})
                })
                seen.add(key)

    return {
        "entities": list(merged_entities.values()),
        "relationships": merged_relationships,
        "entity_count": len(merged_entities),
        "relationship_count": len(merged_relationships)
    }

def get_entity_statistics(merged_graph: Dict) -> Dict:
    entity_type_counts = {}
    relationship_type_counts = {}

    for e in merged_graph["entities"]:
        entity_type_counts[e["type"]] = entity_type_counts.get(e["type"], 0) + 1

    for r in merged_graph["relationships"]:
        relationship_type_counts[r["type"]] = relationship_type_counts.get(r["type"], 0) + 1

    most_mentioned = sorted(
        merged_graph["entities"],
        key=lambda x: x.get("mentions", 0),
        reverse=True
    )[:10]

    return {
        "total_entities": merged_graph["entity_count"],
        "total_relationships": merged_graph["relationship_count"],
        "entity_types": entity_type_counts,
        "relationship_types": relationship_type_counts,
        "most_mentioned_entities": [
            {
                "name": e["name"],
                "type": e["type"],
                "mentions": e["mentions"]
            }
            for e in most_mentioned
        ]
    }
