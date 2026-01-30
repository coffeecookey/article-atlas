import tiktoken
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

# textchunk data model can be used later

# change model to gemini later
def get_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    return len(tokens)


def chunk_text(cleaned_text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    if not cleaned_text:
        return []
    
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        model_name="gpt-3.5-turbo",
        encoding_name="cl100k_base"
    )
    
    # list of token based text chunks (order preserved, no positional metadata)
    raw_chunks = text_splitter.split_text(cleaned_text)
    
    # Final structured chunk list with metadata
    chunks = []
    # to track search position within original text
    current_pos = 0
    
    # i is idx 
    for i, chunk_text in enumerate(raw_chunks):
        start_char = cleaned_text.find(chunk_text, current_pos)
        if start_char == -1:
            start_char = current_pos
        
        end_char = start_char + len(chunk_text)
        
        token_count = get_token_count(chunk_text)
        
        overlap_prev = 0
        if i > 0:
            prev_chunk = chunks[i-1]
            prev_end = prev_chunk['end_char']
            if start_char < prev_end:
                overlap_prev = prev_end - start_char
        
        overlap_next = 0
        if i < len(raw_chunks) - 1:
            next_start_preview = cleaned_text.find(raw_chunks[i+1], end_char - overlap)
            if next_start_preview != -1 and next_start_preview < end_char:
                overlap_next = end_char - next_start_preview
        
        chunk = {
            'chunk_id': f"chunk_{i:04d}",
            'text': chunk_text,
            'token_count': token_count,
            'position': i,
            'total_chunks': len(raw_chunks),
            'start_char': start_char,
            'end_char': end_char,
            'overlap_with_previous': overlap_prev,
            'overlap_with_next': overlap_next
        }
        
        chunks.append(chunk)
        current_pos = start_char + len(chunk_text) - overlap
    
    for chunk in chunks:
        chunk['total_chunks'] = len(chunks)
    
    return chunks


def validate_chunks(chunks: List[Dict], max_tokens: int = 8000, min_overlap: int = 0, max_overlap: int = 200) -> bool:
    if not chunks:
        return False
    
    for chunk in chunks:
        if chunk['token_count'] > max_tokens:
            return False
        
        if chunk['overlap_with_previous'] < min_overlap and chunk['position'] > 0:
            return False
        
        if chunk['overlap_with_previous'] > max_overlap:
            return False
        
        if chunk['overlap_with_next'] > max_overlap:
            return False
    
    for i, chunk in enumerate(chunks):
        if chunk['position'] != i:
            return False
        
        if chunk['total_chunks'] != len(chunks):
            return False
    
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        if current_chunk['end_char'] < next_chunk['start_char']:
            pass
        elif current_chunk['end_char'] > next_chunk['end_char']:
            return False
    
    return True


def chunk_text_semantic(cleaned_text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    if not cleaned_text:
        return []
    
    separators = [
        "\n\n\n",
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        ""
    ]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
        keep_separator=True,
        is_separator_regex=False
    )
    
    raw_chunks = text_splitter.split_text(cleaned_text)
    
    chunks = []
    current_pos = 0
    
    for i, chunk_text in enumerate(raw_chunks):
        start_char = cleaned_text.find(chunk_text, current_pos)
        if start_char == -1:
            start_char = current_pos
        
        end_char = start_char + len(chunk_text)
        
        token_count = get_token_count(chunk_text)
        
        overlap_prev = 0
        if i > 0:
            prev_chunk = chunks[i-1]
            prev_end = prev_chunk['end_char']
            if start_char < prev_end:
                overlap_prev = prev_end - start_char
        
        overlap_next = 0
        
        chunk = {
            'chunk_id': f"chunk_{i:04d}",
            'text': chunk_text,
            'token_count': token_count,
            'position': i,
            'total_chunks': len(raw_chunks),
            'start_char': start_char,
            'end_char': end_char,
            'overlap_with_previous': overlap_prev,
            'overlap_with_next': overlap_next,
            'is_semantic': True
        }
        
        chunks.append(chunk)
        current_pos = start_char + len(chunk_text) - overlap
    
    for chunk in chunks:
        chunk['total_chunks'] = len(chunks)
    
    return chunks


def merge_small_chunks(chunks: List[Dict], min_token_size: int = 100) -> List[Dict]:
    if not chunks:
        return []
    
    merged = []
    current_merged = None
    
    for chunk in chunks:
        if chunk['token_count'] < min_token_size:
            if current_merged is None:
                current_merged = chunk.copy()
            else:
                current_merged['text'] += " " + chunk['text']
                current_merged['token_count'] = get_token_count(current_merged['text'])
                current_merged['end_char'] = chunk['end_char']
        else:
            if current_merged is not None:
                merged.append(current_merged)
                current_merged = None
            merged.append(chunk)
    
    if current_merged is not None:
        merged.append(current_merged)
    
    for i, chunk in enumerate(merged):
        chunk['position'] = i
        chunk['total_chunks'] = len(merged)
        chunk['chunk_id'] = f"chunk_{i:04d}"
    
    return merged


def get_chunk_stats(chunks: List[Dict]) -> Dict:
    if not chunks:
        return {
            'total_chunks': 0,
            'total_tokens': 0,
            'avg_tokens_per_chunk': 0,
            'min_tokens': 0,
            'max_tokens': 0,
            'total_characters': 0,
            'avg_overlap': 0
        }
    
    token_counts = [chunk['token_count'] for chunk in chunks]
    overlaps = [chunk['overlap_with_previous'] for chunk in chunks if chunk['position'] > 0]
    
    return {
        'total_chunks': len(chunks),
        'total_tokens': sum(token_counts),
        'avg_tokens_per_chunk': sum(token_counts) / len(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts),
        'total_characters': sum(len(chunk['text']) for chunk in chunks),
        'avg_overlap': sum(overlaps) / len(overlaps) if overlaps else 0
    }