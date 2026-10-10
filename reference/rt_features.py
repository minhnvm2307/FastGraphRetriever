"""Excerpt of retrieve/src/dataset/retriever.py (SubgraphRAG research repo, branch rel_rag @ 488f574 + local changes, 2026-10-11).

Only the relation-topic (RT) feature code used by the WL checkpoint: relation_topic_source=gte, relation_score_mode=hybrid,
alpha=0.5, calibration=none, use_relation_topic_mask=True, topk_rt=5, relation_topic_cache=None (features computed live).
The live branch is RetrieverDataset._assembly (source in {gte, path_memory_hybrid}, no cache), copied at the bottom.
"""
import string
import torch
import torch.nn.functional as F

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


# ---- RetrieverDataset static helpers (verbatim) ----
class RTHelpers:
    @staticmethod
    def _select_relation_score(score_mode, semantic, lexical, alpha, num_relations):
        """Return the canonical relation score used by DDE and selection."""
        if score_mode == 'semantic':
            return semantic
        if score_mode == 'lexical':
            return lexical
        if score_mode == 'hybrid':
            return alpha * lexical + (1.0 - alpha) * semantic
        if score_mode == 'none':
            return torch.zeros(num_relations, dtype=torch.float32)
        raise ValueError(f"Unknown relation_score_mode: {score_mode}")

    @staticmethod
    def _calibrate_relation_score(
        score: torch.Tensor,
        calibration_mode: str = 'none',
        platt_a: float = 9.77,
        platt_b: float = -5.20,
        power_gamma: float = 1.0,
    ) -> torch.Tensor:
        """Calibrate continuous relation scores to expand contrast between Gold and Noise."""
        if calibration_mode == 'platt':
            return torch.sigmoid(platt_a * score + platt_b)
        elif calibration_mode == 'power':
            return torch.clamp(score, min=0.0) ** power_gamma
        elif calibration_mode == 'none':
            return score
        else:
            raise ValueError(f"Unknown relation_score_calibration: {calibration_mode}")

    @staticmethod
    def _select_relation_topic_mask(
        score: torch.Tensor,
        topk_rt: int | None = None,
        mask_mode: str = 'topk',
        cluster_ratio: float = 0.90,
        min_k: int = 2,
        max_k: int = 15,
    ) -> torch.Tensor:
        """Adaptive cluster or Top-k selection for relation-topic sources."""
        mask = torch.zeros_like(score)
        if score.numel() == 0:
            return mask
        if mask_mode == 'cluster':
            order = torch.argsort(score, descending=True)
            sorted_scores = score[order]
            cutoff = min(min_k, len(sorted_scores))
            for i in range(min_k - 1, min(len(sorted_scores) - 1, max_k)):
                if sorted_scores[i] > 0 and (sorted_scores[i + 1] / sorted_scores[i]) >= cluster_ratio:
                    cutoff = i + 2
                else:
                    break
            mask[order[:cutoff]] = 1.0
            return mask
        else:
            if not isinstance(topk_rt, int) or topk_rt < 1:
                raise ValueError("topk_rt must be a positive int when using a relation-topic mask")
            mask[torch.topk(score, min(topk_rt, score.numel())).indices] = 1.0
            return mask


# ---- live branch of RetrieverDataset._assembly (verbatim, for reference) ----
#                     emb_i = emb_dict[sample_i_id]
#                     semantic = None
#                     lexical = None
#                     if score_mode in {'semantic', 'hybrid'}:
#                         q_emb_i = emb_i['q_emb'].to(torch.float32)  # shape: (1, 1024)
#                         rel_embs_i = emb_i['relation_embs'].to(torch.float32) # shape: (num_relations, 1024)
#                         semantic = torch.matmul(rel_embs_i, q_emb_i.T).squeeze(1)
#                     if score_mode in {'lexical', 'hybrid'}:
#                         lexical = torch.tensor(
#                             [compute_lexical_score(clean_text(sample_i['question']), rel)
#                              for rel in sample_i['relation_list']], dtype=torch.float32
#                         )
#                     raw_score = self._select_relation_score(score_mode, semantic, lexical, score_alpha, num_relations)
#                     score = self._calibrate_relation_score(raw_score, calib_mode, power_gamma=power_gamma)
#                     mask = (
#                         self._select_relation_topic_mask(
#                             score, topk_rt, mask_mode=mask_mode,
#                             cluster_ratio=cluster_ratio, min_k=min_k, max_k=max_k
#                         )
#                         if use_mask else torch.zeros(num_relations)
#                     )
#                     relation_topic_features = torch.stack([score, mask], dim=1)

# ---- topic one-hot (verbatim, for reference) ----
#             num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
#             topic_entity_mask = torch.zeros(num_entities_i)
#             topic_entity_mask[sample_i['q_entity_id_list']] = 1.
#             topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
#             sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()
