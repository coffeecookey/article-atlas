import uuid
import time
import logging
import functools
from typing import Callable, Any, Dict
from datetime import datetime
from collections import deque

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_id(prefix: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    if prefix:
        return f"{prefix}_{timestamp}_{unique_id}"
    return f"{timestamp}_{unique_id}"


def calculate_token_count(text: str, model: str = "gpt-4") -> int:
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"tiktoken failed, using approximation: {e}")
    
    words = text.split()
    return int(len(words) * 1.3)


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }
    
    logger.error(
        f"Error occurred: {error_data['error_type']}: {error_data['error_message']}"
    )
    
    if context:
        logger.error(f"Context: {context}")
    
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")


def rate_limiter_decorator(calls_per_minute: int):
    def decorator(func: Callable) -> Callable:
        call_times = deque()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            while call_times and call_times[0] < now - 60:
                call_times.popleft()
            
            if len(call_times) >= calls_per_minute:
                sleep_time = 60 - (now - call_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    call_times.popleft()
            
            call_times.append(time.time())
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_url(url: str) -> bool:
    import re
    
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def sanitize_filename(filename: str) -> str:
    import re
    
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename.strip('_')
    
    return filename[:200]


def format_timestamp(dt: datetime = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)


def batch_items(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


def measure_execution_time(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper


def calculate_similarity(text1: str, text2: str) -> float:
    from difflib import SequenceMatcher
    
    if not text1 or not text2:
        return 0.0
    
    matcher = SequenceMatcher(None, text1.lower(), text2.lower())
    return matcher.ratio()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def ensure_directory_exists(directory_path: str) -> None:
    import os
    os.makedirs(directory_path, exist_ok=True)


def get_file_size_mb(filepath: str) -> float:
    import os
    
    if not os.path.exists(filepath):
        return 0.0
    
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)


def merge_dictionaries(*dicts) -> dict:
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def chunk_list(items: list, chunk_size: int) -> list:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_list(nested_list: list) -> list:
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def remove_duplicates(items: list, key: Callable = None) -> list:
    if key is None:
        return list(dict.fromkeys(items))
    
    seen = set()
    result = []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def parse_json_safe(json_string: str, default: Any = None) -> Any:
    import json
    
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def get_nested_value(data: dict, keys: list, default: Any = None) -> Any:
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def timer_context():
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, *args):
            self.end = time.time()
            self.elapsed = self.end - self.start
            logger.info(f"Operation took {self.elapsed:.2f} seconds")
    
    return Timer()