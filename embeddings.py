"""
Embedding generation and semantic similarity engine with TF-IDF fallback.
"""
import logging
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_API_KEY, DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def build_candidate_search_text(candidate: "CandidateProfile") -> str:
    """Combines candidate attributes into a rich text block for embedding and TF-IDF."""
    parts = []
    if candidate.headline:
        parts.append(f"Headline: {candidate.headline}")
    if candidate.job_titles:
        parts.append(f"Job Titles: {', '.join(candidate.job_titles)}")
    if candidate.skills:
        parts.append(f"Skills: {', '.join(candidate.skills)}")
    if candidate.experience_years > 0:
        parts.append(f"Experience: {candidate.experience_years} years")

    # Add work experience summaries
    for exp in candidate.work_experience:
        parts.append(f"Role: {exp.title} at {exp.company}. {exp.description}")

    # Add education
    for edu in candidate.education:
        parts.append(f"Education: {edu.degree} in {edu.field_of_study} from {edu.institution}")

    if candidate.projects:
        parts.append(f"Projects: {', '.join(candidate.projects)}")

    full_text = "\n".join(parts)
    if not full_text.strip():
        full_text = candidate.raw_text[:2000]
    return full_text


class SemanticSearchEngine:
    """Handles semantic similarity search using OpenAI Embeddings or TF-IDF fallback."""

    def __init__(self):
        self.use_openai = bool(OPENAI_API_KEY)
        self.openai_client = None
        if self.use_openai:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client for embeddings: {e}")
                self.use_openai = False

    def compute_similarity(self, query_text: str, candidates: List["CandidateProfile"]) -> Dict[str, float]:
        """
        Computes cosine similarity between query/JD text and candidate texts.
        Returns a dictionary mapping candidate ID -> similarity score (0.0 to 1.0).
        """
        if not candidates or not query_text.strip():
            return {c.id: 0.0 for c in candidates}

        candidate_texts = [build_candidate_search_text(c) for c in candidates]

        if self.use_openai and self.openai_client:
            try:
                return self._compute_openai_similarity(query_text, candidate_texts, candidates)
            except Exception as e:
                logger.warning(f"OpenAI embedding computation failed, falling back to TF-IDF: {e}")

        # Deterministic TF-IDF Fallback
        return self._compute_tfidf_similarity(query_text, candidate_texts, candidates)

    def _compute_tfidf_similarity(self, query_text: str, candidate_texts: List[str], candidates: List["CandidateProfile"]) -> Dict[str, float]:
        """TF-IDF vectorizer + Cosine Similarity computation."""
        corpus = [query_text] + candidate_texts
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = tfidf_matrix[0:1]
            cand_vecs = tfidf_matrix[1:]
            sim_scores = cosine_similarity(query_vec, cand_vecs)[0]

            results = {}
            for idx, c in enumerate(candidates):
                # Normalize score safely to 0-1
                results[c.id] = float(max(0.0, min(1.0, sim_scores[idx])))
            return results
        except Exception as e:
            logger.error(f"TF-IDF similarity failed: {e}")
            return {c.id: 0.0 for c in candidates}

    def _compute_openai_similarity(self, query_text: str, candidate_texts: List[str], candidates: List["CandidateProfile"]) -> Dict[str, float]:
        """OpenAI Embedding similarity calculation."""
        all_texts = [query_text] + candidate_texts
        response = self.openai_client.embeddings.create(
            input=all_texts,
            model=DEFAULT_EMBEDDING_MODEL
        )
        embeddings = [data.embedding for data in response.data]
        query_emb = np.array(embeddings[0]).reshape(1, -1)
        cand_embs = np.array(embeddings[1:])

        # Cosine similarity
        norm_query = query_emb / np.linalg.norm(query_emb)
        norm_cands = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True)
        sims = np.dot(norm_query, norm_cands.T)[0]

        results = {}
        for idx, c in enumerate(candidates):
            results[c.id] = float(max(0.0, min(1.0, sims[idx])))
        return results
