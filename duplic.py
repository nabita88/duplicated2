import re
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein
from functools import lru_cache


@dataclass
class DuplicationMetrics:
    original_count: int
    deduplicated_count: int
    compression_ratio: float
    information_retention: float
    semantic_diversity: float
    processing_time: float


class DeduplicatorConfig:
    DEFAULT_CONFIG = {
        'sentence_model': 'jhgan/ko-sroberta-multitask',
        'surface_threshold': 0.95,
        'semantic_threshold': 0.9,
        'context_threshold': 0.85,
        'min_sentence_length': 5,
        'context_window': 2,
        'use_gpu': torch.cuda.is_available()
    }

    @classmethod
    def get_config(cls, custom_config: Dict = None) -> Dict:
        config = cls.DEFAULT_CONFIG.copy()
        if custom_config:
            config.update(custom_config)
        return config


class SemanticDeduplicator:

    def __init__(self, config: Dict = None):
        self.config = DeduplicatorConfig.get_config(config)

        self.sentence_encoder = SentenceTransformer(
            self.config['sentence_model']
        )

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2)
        )

        self.embedding_cache = {}
        self.similarity_cache = {}

        self._compile_patterns()

    def _compile_patterns(self):
        self.patterns = {
            'sentence_split': re.compile(r'(?<=[.!?])\s+(?=[A-Z가-힣])'),
            'normalize': re.compile(r'[^가-힣a-zA-Z0-9\s]'),
            'whitespace': re.compile(r'\s+'),
            'html_entity': re.compile(r'&[a-zA-Z]+;'),
            'number': re.compile(r'\d+'),
        }

    def deduplicate(self, text: str) -> Tuple[str, DuplicationMetrics]:
        import time
        start_time = time.time()

        sentences = self._split_sentences(text)
        original_count = len(sentences)

        if original_count == 0:
            return "", DuplicationMetrics(0, 0, 0, 0, 0, 0)

        level1_unique = self._surface_level_dedup(sentences)
        level2_unique = self._semantic_level_dedup(level1_unique)
        level3_unique = self._contextual_level_dedup(level2_unique)

        deduplicated_text = ' '.join(level3_unique)

        metrics = self._calculate_metrics(
            sentences, level3_unique, time.time() - start_time
        )

        return deduplicated_text, metrics

    def _split_sentences(self, text: str) -> List[str]:
        text = self._preprocess_text(text)
        sentences = self._primary_sentence_split(text)

        if self._needs_additional_splitting(sentences):
            sentences = self._additional_sentence_split(sentences)

        return self._filter_valid_sentences(sentences)

    def _preprocess_text(self, text: str) -> str:
        return self.patterns['html_entity'].sub(' ', text)

    def _primary_sentence_split(self, text: str) -> List[str]:
        return self.patterns['sentence_split'].split(text)

    def _needs_additional_splitting(self, sentences: List[str]) -> bool:
        return len(sentences) == 1 and len(sentences[0]) > 100

    def _additional_sentence_split(self, sentences: List[str]) -> List[str]:
        text = sentences[0]

        sentences = self._split_by_keywords(text)

        if len(sentences) == 1:
            sentences = self._force_split_by_length(text, chunk_size=50)

        return sentences

    def _split_by_keywords(self, text: str) -> List[str]:
        alt_pattern = re.compile(
            r'(?<=지원)(?=[A-Z가-힣])|(?<=수행)(?=[A-Z가-힣])|(?<=프로젝트)(?=[A-Z가-힣])'
        )
        return alt_pattern.split(text)

    def _force_split_by_length(self, text: str, chunk_size: int) -> List[str]:
        sentences = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                sentences.append(chunk)
        return sentences

    def _filter_valid_sentences(self, sentences: List[str]) -> List[str]:
        return [
            s.strip() for s in sentences
            if len(s.strip()) >= self.config['min_sentence_length']
        ]

    def _surface_level_dedup(self, sentences: List[str]) -> List[str]:
        unique = []
        seen_hashes = set()
        seen_normalized = {}

        for sent in sentences:
            normalized = self._normalize_text(sent)

            sent_hash = hashlib.md5(normalized.encode()).hexdigest()

            if sent_hash in seen_hashes:
                continue

            is_duplicate = self._check_surface_similarity(
                sent, normalized, seen_normalized, unique
            )

            if not is_duplicate:
                unique.append(sent)
                seen_hashes.add(sent_hash)
                seen_normalized[normalized] = sent

        return unique

    def _check_surface_similarity(self, sent: str, normalized: str,
                                  seen_normalized: Dict, unique: List[str]) -> bool:
        for norm_text, original in seen_normalized.items():
            similarity = self._calculate_edit_similarity(normalized, norm_text)
            if similarity > self.config['surface_threshold']:
                if len(sent) > len(original):
                    unique.remove(original)
                    unique.append(sent)
                    seen_normalized[normalized] = sent
                return True
        return False

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = self.patterns['normalize'].sub('', text)
        text = self.patterns['number'].sub('NUM', text)
        text = self.patterns['whitespace'].sub(' ', text)

        return text.strip()

    def _calculate_edit_similarity(self, text1: str, text2: str) -> float:
        distance = Levenshtein.distance(text1, text2)
        max_len = max(len(text1), len(text2))

        if max_len == 0:
            return 1.0

        return 1.0 - (distance / max_len)

    def _semantic_level_dedup(self, sentences: List[str]) -> List[str]:
        if len(sentences) <= 1:
            return sentences

        embeddings = self._get_embeddings(tuple(sentences))

        similarity_matrix = cosine_similarity(embeddings)

        duplicate_groups = self._find_duplicate_groups(
            similarity_matrix,
            self.config['semantic_threshold']
        )

        unique = self._extract_representatives_from_groups(
            duplicate_groups, sentences, embeddings
        )

        return unique

    def _extract_representatives_from_groups(self,
                                             duplicate_groups: List[List[int]],
                                             sentences: List[str],
                                             embeddings: np.ndarray) -> List[str]:
        unique = []
        for group_indices in duplicate_groups:
            group_sentences = [sentences[i] for i in group_indices]
            representative = self._select_representative(
                group_sentences,
                [embeddings[i] for i in group_indices]
            )
            unique.append(representative)
        return unique

    @lru_cache(maxsize=1000)
    def _get_embeddings(self, sentences: Tuple[str]) -> np.ndarray:
        if isinstance(sentences, list):
            sentences = tuple(sentences)

        embeddings = self.sentence_encoder.encode(
            list(sentences),
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings

    def _find_duplicate_groups(self,
                               similarity_matrix: np.ndarray,
                               threshold: float) -> List[List[int]]:
        n = len(similarity_matrix)
        visited = set()
        groups = []

        for i in range(n):
            if i in visited:
                continue

            group = [i]
            visited.add(i)

            for j in range(i + 1, n):
                if j not in visited and similarity_matrix[i][j] > threshold:
                    group.append(j)
                    visited.add(j)

            groups.append(group)

        return groups

    def _select_representative(self,
                               sentences: List[str],
                               embeddings: List[np.ndarray]) -> str:
        if len(sentences) == 1:
            return sentences[0]

        scores = self._calculate_representation_scores(sentences, embeddings)
        best_idx = np.argmax(scores)
        return sentences[best_idx]

    def _calculate_representation_scores(self,
                                         sentences: List[str],
                                         embeddings: List[np.ndarray]) -> List[float]:
        embeddings_array = np.array(embeddings)

        centrality = self._calculate_centrality(embeddings_array)

        length_scores = self._calculate_length_scores(sentences)

        scores = []
        for i in range(len(sentences)):
            score = 0.4 * length_scores[i] + 0.6 * centrality[i]
            scores.append(score)

        return scores

    def _calculate_centrality(self, embeddings: np.ndarray) -> np.ndarray:
        similarities = cosine_similarity(embeddings)
        return similarities.mean(axis=1)

    def _calculate_length_scores(self, sentences: List[str]) -> List[float]:
        max_length = max(len(s) for s in sentences)
        return [len(s) / max_length for s in sentences]

    def _contextual_level_dedup(self, sentences: List[str]) -> List[str]:
        if len(sentences) <= 1:
            return sentences

        unique = []
        context_embeddings = []

        for i, sent in enumerate(sentences):
            context_window = self._get_context_window(sentences, i)

            context_emb = self._create_contextual_embedding(
                sent, context_window, position=i, total=len(sentences)
            )

            is_redundant = self._check_contextual_redundancy(
                context_emb, context_embeddings
            )

            if not is_redundant:
                unique.append(sent)
                context_embeddings.append(context_emb)

        return unique

    def _get_context_window(self, sentences: List[str], index: int) -> List[str]:
        window_size = self.config['context_window']
        start_idx = max(0, index - window_size)
        end_idx = min(len(sentences), index + window_size + 1)
        return sentences[start_idx:end_idx]

    def _check_contextual_redundancy(self,
                                     context_emb: np.ndarray,
                                     existing_embeddings: List[np.ndarray]) -> bool:
        for existing_emb in existing_embeddings:
            similarity = self._contextual_similarity(context_emb, existing_emb)
            if similarity > self.config['context_threshold']:
                return True
        return False

    def _create_contextual_embedding(self,
                                     sentence: str,
                                     context_window: List[str],
                                     position: int,
                                     total: int) -> np.ndarray:
        sent_emb = self._get_sentence_embedding(sentence)
        context_emb = self._get_context_embedding(context_window, sent_emb)
        position_encoding = self._create_position_encoding(position, sent_emb.shape[0])

        return self._combine_embeddings(sent_emb, context_emb, position_encoding)

    def _get_sentence_embedding(self, sentence: str) -> np.ndarray:
        return self.sentence_encoder.encode(sentence, convert_to_numpy=True)

    def _get_context_embedding(self, context_window: List[str],
                               default_emb: np.ndarray) -> np.ndarray:
        if len(context_window) > 1:
            context_text = ' '.join(context_window)
            return self.sentence_encoder.encode(context_text, convert_to_numpy=True)
        return default_emb

    def _create_position_encoding(self, position: int, dim: int) -> np.ndarray:
        return np.array([
            np.sin(position / 10000 ** (i / dim))
            if i % 2 == 0 else
            np.cos(position / 10000 ** ((i - 1) / dim))
            for i in range(dim)
        ])

    def _combine_embeddings(self, sent_emb: np.ndarray,
                            context_emb: np.ndarray,
                            position_encoding: np.ndarray) -> np.ndarray:
        return np.concatenate([
            sent_emb * 0.5,
            context_emb * 0.3,
            position_encoding * 0.2
        ])

    def _contextual_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cos_sim)

    def _calculate_metrics(self,
                           original: List[str],
                           deduplicated: List[str],
                           processing_time: float) -> DuplicationMetrics:
        original_count = len(original)
        dedup_count = len(deduplicated)

        compression_ratio = self._calculate_compression_ratio(original_count, dedup_count)

        information_retention = self._calculate_information_retention(original, deduplicated)

        semantic_diversity = self._calculate_semantic_diversity(deduplicated)

        return DuplicationMetrics(
            original_count=original_count,
            deduplicated_count=dedup_count,
            compression_ratio=compression_ratio,
            information_retention=information_retention,
            semantic_diversity=semantic_diversity,
            processing_time=processing_time
        )

    def _calculate_compression_ratio(self, original_count: int, dedup_count: int) -> float:
        if original_count > 0:
            return dedup_count / original_count
        return 0

    def _calculate_information_retention(self,
                                         original: List[str],
                                         deduplicated: List[str]) -> float:
        original_text = ' '.join(original)
        dedup_text = ' '.join(deduplicated)

        original_tokens = set(original_text.lower().split())
        dedup_tokens = set(dedup_text.lower().split())

        if original_tokens:
            return len(dedup_tokens & original_tokens) / len(original_tokens)
        return 0

    def _calculate_semantic_diversity(self, deduplicated: List[str]) -> float:
        if len(deduplicated) > 1:
            dedup_embeddings = self._get_embeddings(tuple(deduplicated))
            diversity_matrix = 1 - cosine_similarity(dedup_embeddings)
            return diversity_matrix.mean()
        return 0


class FastSemanticDeduplicator(SemanticDeduplicator):

    def __init__(self, config: Dict = None):
        super().__init__(config)

        self.num_perm = 128
        self.threshold = 0.9

    def deduplicate_batch(self, texts: List[str]) -> List[Tuple[str, DuplicationMetrics]]:
        results = []

        signatures = self._create_minhash_signatures(texts)

        candidate_pairs = self._lsh_candidate_selection(signatures)

        for i, text in enumerate(texts):
            candidates = [j for j in candidate_pairs.get(i, [])]

            if candidates:
                is_duplicate = self._verify_duplicate(text, [texts[j] for j in candidates])

                if not is_duplicate:
                    dedup_text, metrics = self.deduplicate(text)
                    results.append((dedup_text, metrics))
            else:
                dedup_text, metrics = self.deduplicate(text)
                results.append((dedup_text, metrics))

        return results

    def _create_minhash_signatures(self, texts: List[str]) -> List[List[int]]:
        return [[hash(text) for _ in range(self.num_perm)] for text in texts]

    def _lsh_candidate_selection(self, signatures: List[List[int]]) -> Dict[int, Set[int]]:
        candidates = defaultdict(set)
        return candidates

    def _verify_duplicate(self, text: str, candidates: List[str]) -> bool:
        text_emb = self.sentence_encoder.encode(text, convert_to_numpy=True)

        for candidate in candidates:
            cand_emb = self.sentence_encoder.encode(candidate, convert_to_numpy=True)
            similarity = cosine_similarity([text_emb], [cand_emb])[0][0]

            if similarity > self.config['semantic_threshold']:
                return True

        return False


if __name__ == "__main__":
    import sys
    import time

    sample_text = """
    UniERP 시스템 개발 및 운영고객 요청을 서비스데스크로 접수 후 요청사항 지원UniERP 변경 요청에 대한설계/개발 지원UniERP 개선 프로젝트 수행(설계/개발)판매 계획/실적 분석 등 운영 개선 프로젝트
    """

    deduplicator = SemanticDeduplicator()
    cleaned_text, metrics = deduplicator.deduplicate(sample_text)

    print(f"정제된 텍스트:{cleaned_text}")