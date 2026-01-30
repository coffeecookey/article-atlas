import re
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict
from rapidfuzz import fuzz, process

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("SentenceTransformers not installed. Install with: pip install sentence-transformers")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not installed. Install with: pip install spacy")

FUZZY_THRESHOLD = 85
SEMANTIC_THRESHOLD = 0.80

COMMON_ABBREVIATIONS = {
    "US": "United States",
    "USA": "United States",
    "UK": "United Kingdom",
    "EU": "European Union",
    "UN": "United Nations",
    "NASA": "National Aeronautics and Space Administration",
    "FBI": "Federal Bureau of Investigation",
    "CIA": "Central Intelligence Agency",
    "WHO": "World Health Organization",
    "GDP": "Gross Domestic Product",
    "CEO": "Chief Executive Officer",
    "CTO": "Chief Technology Officer",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
}

_sentence_model = None
_spacy_nlp = None


def get_sentence_model():
    global _sentence_model
    if not SEMANTIC_AVAILABLE:
        return None
    
    if _sentence_model is None:
        try:
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded SentenceTransformer model")
        except Exception as e:
            print(f"Failed to load SentenceTransformer: {e}")
            return None
    
    return _sentence_model


def get_spacy_model():
    global _spacy_nlp
    if not SPACY_AVAILABLE:
        return None
    
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
            print("Loaded spaCy model")
        except OSError:
            print("spaCy model not found. Download with: python -m spacy download en_core_web_sm")
            return None
    
    return _spacy_nlp

Entity = Dict[str, Any]
Relationship = Dict[str, Any]

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def expand_abbreviation(text: str) -> str:
    return COMMON_ABBREVIATIONS.get(text.upper(), text)


def calculate_fuzzy_similarity(name1: str, name2: str) -> float:
    norm1 = normalize_text(name1)
    norm2 = normalize_text(name2)
    
    ratio = fuzz.ratio(norm1, norm2)
    partial_ratio = fuzz.partial_ratio(norm1, norm2)
    token_sort = fuzz.token_sort_ratio(norm1, norm2)
    
    return max(ratio, partial_ratio, token_sort)


def calculate_semantic_similarity(name1: str, name2: str) -> float:
    model = get_sentence_model()
    if model is None:
        return calculate_fuzzy_similarity(name1, name2) / 100.0
    
    try:
        embeddings = model.encode([name1, name2])
        
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    except Exception as e:
        print(f"Semantic similarity failed: {e}")
        return calculate_fuzzy_similarity(name1, name2) / 100.0


def are_entities_similar(entity1: Entity, entity2: Entity) -> bool:
    if entity1.get("type") != entity2.get("type"):
        return False
    
    name1 = entity1.get("name", "")
    name2 = entity2.get("name", "")
    
    if normalize_text(name1) == normalize_text(name2):
        return True
    
    if expand_abbreviation(name1) == name2 or expand_abbreviation(name2) == name1:
        return True
    
    fuzzy_score = calculate_fuzzy_similarity(name1, name2)
    if fuzzy_score >= FUZZY_THRESHOLD:
        return True
    
    if SEMANTIC_AVAILABLE:
        semantic_score = calculate_semantic_similarity(name1, name2)
        if semantic_score >= SEMANTIC_THRESHOLD:
            return True
    
    return False


def normalize_entities(entities: List[Entity]) -> List[Entity]:
    if not entities:
        return []
    
    print(f"\n{'='*70}")
    print(f"NORMALIZING {len(entities)} ENTITIES")
    print(f"{'='*70}")
    
    entities_by_type = defaultdict(list)
    for entity in entities:
        entity_type = entity.get("type", "UNKNOWN")
        entities_by_type[entity_type].append(entity)
    
    normalized = []
    merge_count = 0
    
    for entity_type, type_entities in entities_by_type.items():
        print(f"\nProcessing {len(type_entities)} {entity_type} entities...")
        
        merged_indices = set()
        
        for i, entity1 in enumerate(type_entities):
            if i in merged_indices:
                continue
            
            merge_group = [entity1]
            merge_group_indices = {i}
            
            for j, entity2 in enumerate(type_entities[i+1:], start=i+1):
                if j in merged_indices:
                    continue
                
                if are_entities_similar(entity1, entity2):
                    merge_group.append(entity2)
                    merge_group_indices.add(j)
            
            merged_indices.update(merge_group_indices)
            
            if len(merge_group) > 1:
                merge_count += len(merge_group) - 1
                print(f"  Merging {len(merge_group)} similar entities: {[e.get('name') for e in merge_group]}")
            
            merged_entity = merge_entity_group(merge_group)
            normalized.append(merged_entity)
    
    print(f"\n Normalization complete!")
    print(f"  Original entities: {len(entities)}")
    print(f"  Normalized entities: {len(normalized)}")
    print(f"  Merged: {merge_count} entities")
    
    return normalized


def merge_entity_group(entities: List[Entity]) -> Entity:
    if len(entities) == 1:
        return entities[0]
    
    name_counts = defaultdict(int)
    for entity in entities:
        name = entity.get("name", "")
        mentions = entity.get("mentions", 1)
        name_counts[name] += mentions
    
    canonical_name = max(name_counts.items(), key=lambda x: x[1])[0]
    
    all_names = {e.get("name", "") for e in entities}
    aliases = list(all_names - {canonical_name})
    
    merged_properties = {}
    for entity in entities:
        merged_properties.update(entity.get("properties", {}))
    
    all_chunk_ids = []
    for entity in entities:
        all_chunk_ids.extend(entity.get("chunk_ids", []))
    unique_chunk_ids = list(dict.fromkeys(all_chunk_ids))
    
    total_mentions = sum(e.get("mentions", 1) for e in entities)
    
    entity_id = entities[0].get("id", f"{normalize_text(canonical_name)}_{entities[0].get('type', '').lower()}")
    
    return {
        "id": entity_id,
        "name": canonical_name,
        "type": entities[0].get("type", "UNKNOWN"),
        "properties": merged_properties,
        "mentions": total_mentions,
        "chunk_ids": unique_chunk_ids,
        "canonical_name": canonical_name,
        "aliases": aliases
    }


def resolve_coreferences(
    entities: List[Entity],
    relationships: List[Relationship],
    text: Optional[str] = None
) -> Tuple[List[Entity], List[Relationship]]:
    nlp = get_spacy_model()
    
    if nlp is None or not text:
        print("Skipping coreference resolution (spaCy unavailable or no text)")
        return entities, relationships
    
    print(f"\n{'='*70}")
    print(f"RESOLVING COREFERENCES")
    print(f"{'='*70}")
    
    entity_map = {e.get("name", "").lower(): e for e in entities}
    
    for entity in entities:
        for alias in entity.get("aliases", []):
            entity_map[alias.lower()] = entity
    
    doc = nlp(text[:1000000])
    
    coreference_map = {}
    
    for token in doc:
        if token.pos_ == "PRON":
            closest_entity = None
            min_distance = float('inf')
            
            for ent in doc.ents:
                if ent.end < token.i:
                    distance = token.i - ent.end
                    if distance < min_distance:
                        entity_text = ent.text.lower()
                        if entity_text in entity_map:
                            closest_entity = entity_map[entity_text]
                            min_distance = distance
            
            if closest_entity:
                coreference_map[token.text] = closest_entity.get("id")
    
    updated_relationships = []
    update_count = 0
    
    for rel in relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")
        
        new_source = coreference_map.get(source, source)
        new_target = coreference_map.get(target, target)
        
        if new_source != source or new_target != target:
            update_count += 1
        
        updated_relationships.append({
            **rel,
            "source": new_source,
            "target": new_target
        })
    
    print(f"Resolved {update_count} coreferences in relationships")
    
    return entities, updated_relationships


def validate_entity_types(
    entities: List[Entity],
    allowed_types: List[str]
) -> List[Entity]:
    if not allowed_types:
        return entities
    
    print(f"\n{'='*70}")
    print(f"VALIDATING ENTITY TYPES")
    print(f"{'='*70}")
    print(f"Allowed types: {', '.join(allowed_types)}")
    
    allowed_types_set = {t.upper() for t in allowed_types}
    
    valid_entities = []
    invalid_entities = []
    
    for entity in entities:
        entity_type = entity.get("type", "").upper()
        
        if entity_type in allowed_types_set:
            valid_entities.append(entity)
        else:
            invalid_entities.append(entity)
    
    print(f"\n Validation complete!")
    print(f"  Valid entities: {len(valid_entities)}")
    print(f"  Invalid entities: {len(invalid_entities)}")
    
    if invalid_entities:
        print(f"\n  Removed entity types:")
        type_counts = defaultdict(int)
        for e in invalid_entities:
            type_counts[e.get("type", "UNKNOWN")] += 1
        for etype, count in sorted(type_counts.items()):
            print(f"    {etype}: {count}")
    
    return valid_entities


def normalize_pipeline(
    entities: List[Entity],
    relationships: List[Relationship],
    text: Optional[str] = None,
    allowed_types: Optional[List[str]] = None
) -> Tuple[List[Entity], List[Relationship]]:
    print(f"\n{'='*70}")
    print(f"STARTING NORMALIZATION PIPELINE")
    print(f"{'='*70}")
    print(f"Input: {len(entities)} entities, {len(relationships)} relationships")
    
    entities = normalize_entities(entities)
    
    entities, relationships = resolve_coreferences(entities, relationships, text)
    
    if allowed_types:
        entities = validate_entity_types(entities, allowed_types)
    
    print(f"\n{'='*70}")
    print(f"NORMALIZATION PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {len(entities)} entities, {len(relationships)} relationships")
    
    return entities, relationships


def get_entity_statistics(entities: List[Entity]) -> Dict[str, Any]:
    stats = {
        "total_entities": len(entities),
        "entity_types": defaultdict(int),
        "entities_with_aliases": 0,
        "total_aliases": 0,
        "avg_mentions": 0,
        "top_entities": []
    }
    
    for entity in entities:
        stats["entity_types"][entity.get("type", "UNKNOWN")] += 1
        
        aliases = entity.get("aliases", [])
        if aliases:
            stats["entities_with_aliases"] += 1
            stats["total_aliases"] += len(aliases)
    
    if entities:
        stats["avg_mentions"] = sum(e.get("mentions", 1) for e in entities) / len(entities)
        
        sorted_entities = sorted(entities, key=lambda e: e.get("mentions", 1), reverse=True)
        stats["top_entities"] = [
            {
                "name": e.get("name"),
                "type": e.get("type"),
                "mentions": e.get("mentions", 1),
                "aliases": e.get("aliases", [])
            }
            for e in sorted_entities[:10]
        ]
    
    return dict(stats)
