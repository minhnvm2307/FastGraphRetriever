"""Helpers kept from SubgraphRAG's retriever dataset: sharded embedding access and the lexical question--relation score."""
from collections import OrderedDict
import os
import string

import torch


class ShardedEmbDict:
    """Lazy-loading dict-like wrapper for sharded embeddings with LRU cache & mmap."""

    def __init__(self, shard_dir: str, index: dict, max_cache_shards: int = 12):
        self._shard_dir = shard_dir
        self._index = index
        self._max_cache = max_cache_shards
        self._cache = OrderedDict()

    def _load_shard(self, shard_file: str):
        if shard_file in self._cache:
            self._cache.move_to_end(shard_file)
            return self._cache[shard_file]

        shard_path = os.path.join(self._shard_dir, shard_file)
        data = torch.load(shard_path, map_location="cpu", mmap=True)
        if len(self._cache) >= self._max_cache:
            self._cache.popitem(last=False)
        self._cache[shard_file] = data
        return data

    def __contains__(self, key):
        return str(key) in self._index

    def __getitem__(self, key):
        str_key = str(key)
        shard_file = self._index[str_key]
        cache_data = self._load_shard(shard_file)
        if str_key in cache_data:
            return cache_data[str_key]
        try:
            return cache_data[int(str_key)]
        except (KeyError, ValueError):
            raise KeyError(key)


STOPWORDS = {
    'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'is', 'was', 'were', 'are', 'has', 'have', 'had', 'do', 'does', 'did'
}

SYNONYM_GROUPS = [
    {'speak', 'spoken', 'speech', 'language', 'languages'},
    {'born', 'birth', 'originated', 'origin'},
    {'died', 'death', 'deceased', 'burial', 'grave'},
    {'written', 'write', 'author', 'book', 'writer', 'subjects'},
    {'acted', 'actor', 'actress', 'starring', 'play', 'plays', 'cast', 'role', 'roles'},
    {'produced', 'producer'},
    {'directed', 'director'},
    {'created', 'creator', 'creative'},
    {'founded', 'founder', 'founding'},
    {'located', 'location', 'place', 'containedby', 'contains', 'adjoins', 'adjoin'},
    {'member', 'membership', 'participant', 'participated'},
    {'owned', 'owner', 'proprietor'},
    {'married', 'marriage', 'spouse', 'wife', 'husband'},
    {'parent', 'parents', 'children', 'child', 'son', 'daughter', 'father', 'mother'},
    {'nationality', 'citizen', 'country', 'nation'}
]

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [word for word in text.split() if word not in STOPWORDS]

def get_token_similarity(q_token, r_token):
    if q_token == r_token:
        return 1.0
    for group in SYNONYM_GROUPS:
        if q_token in group and r_token in group:
            return 0.9
    min_len = min(len(q_token), len(r_token))
    if min_len >= 4:
        if q_token[:min_len] == r_token[:min_len]:
            return 0.9
        if q_token[:4] == r_token[:4]:
            return 0.8
    return 0.0

def compute_lexical_score(question_tokens, relation_name):
    parts = relation_name.split('.')
    num_parts = len(parts)

    total_score = 0.0
    total_weight = 0.0

    for idx, part in enumerate(parts):
        if idx == num_parts - 1:
            weight = 1.0
        elif idx == num_parts - 2:
            weight = 0.5
        else:
            weight = 0.2

        part_tokens = part.split('_')
        for r_tok in part_tokens:
            if len(r_tok) < 2:
                continue
            r_tok_lower = r_tok.lower()
            if r_tok_lower in STOPWORDS:
                continue
            best_tok_score = 0.0
            for q_tok in question_tokens:
                score = get_token_similarity(q_tok, r_tok_lower)
                if score > best_tok_score:
                    best_tok_score = score
            total_score += best_tok_score * weight
            total_weight += weight

    if total_weight == 0.0:
        return 0.0
    return total_score / total_weight
