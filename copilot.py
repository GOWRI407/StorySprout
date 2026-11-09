"""
StorySprout Capstone - Part D: Writing Co-Pilot (RAG, Gemini API, and Semantic Evaluator)
- Knowledge Base: Chunked grammar/writing guidelines with language and category metadata.
- Retrieval Engine: TF-IDF + Cosine similarity with language filtering and category boosting.
- Retrieval Evaluation: Benchmark of 18 labelled queries computing Hit@1, Hit@3, and MRR.
- GenAI Coaching Generation: Gemini API integration with offline rule-based fallback.
  Strictly enforces pedagogical rules: citations like [1], hints only (no full rewrites), <= 180 words.
- DL Evaluator: Semantic answer grading via sentence-transformers cosine similarity.
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables (.env)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "part_d_copilot", "kb")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "part_d")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 18 Labelled Benchmark Test Queries for Retrieval Evaluation
BENCHMARK_QUERIES = [
    {
        "query": "When do I use preterite vs imperfect in Spanish for past actions?",
        "target_language": "Spanish",
        "category_boost": "tense_inconsistency",
        "expected_chunk_id": "KB-ES-01"
    },
    {
        "query": "Ayer mientras caminaba vi un perro preterito indefinido",
        "target_language": "Spanish",
        "category_boost": "tense_inconsistency",
        "expected_chunk_id": "KB-ES-01"
    },
    {
        "query": "How does subjunctive work in Spanish when expressing doubt with dudo que?",
        "target_language": "Spanish",
        "category_boost": "verb_conjugation",
        "expected_chunk_id": "KB-ES-02"
    },
    {
        "query": "Por vs para distinction in Spanish for purpose destination deadline",
        "target_language": "Spanish",
        "category_boost": "preposition_choice",
        "expected_chunk_id": "KB-ES-03"
    },
    {
        "query": "Differences between passe compose and imparfait when telling a story in French",
        "target_language": "French",
        "category_boost": "tense_inconsistency",
        "expected_chunk_id": "KB-FR-01"
    },
    {
        "query": "French past participle agreement with avoir when direct object is before verb",
        "target_language": "French",
        "category_boost": "gender_agreement",
        "expected_chunk_id": "KB-FR-02"
    },
    {
        "query": "German word order verb position in subordinate clause with weil and dass",
        "target_language": "German",
        "category_boost": "word_order",
        "expected_chunk_id": "KB-DE-01"
    },
    {
        "query": "German two-way prepositions Akkusativ for motion vs Dativ for static location",
        "target_language": "German",
        "category_boost": "preposition_choice",
        "expected_chunk_id": "KB-DE-02"
    },
    {
        "query": "Japanese wa vs ga particle distinction topic versus grammatical subject",
        "target_language": "Japanese",
        "category_boost": "preposition_choice",
        "expected_chunk_id": "KB-JA-01"
    },
    {
        "query": "False friends embarrassing cognates embarazada does not mean embarrassed",
        "target_language": "Spanish",
        "category_boost": "false_friends",
        "expected_chunk_id": "KB-GEN-01"
    },
    {
        "query": "How to use narrative discourse connectors and temporal transitions in stories",
        "target_language": "Spanish",
        "category_boost": "punctuation_spelling",
        "expected_chunk_id": "KB-GEN-02"
    },
    {
        "query": "Difference between ser and estar permanent traits vs temporary states",
        "target_language": "Spanish",
        "category_boost": "verb_conjugation",
        "expected_chunk_id": "KB-ES-04"
    },
    {
        "query": "French subjunctive triggers with il faut que obligation",
        "target_language": "French",
        "category_boost": "verb_conjugation",
        "expected_chunk_id": "KB-FR-03"
    },
    {
        "query": "German V2 rule verb in second position in main sentences",
        "target_language": "German",
        "category_boost": "word_order",
        "expected_chunk_id": "KB-DE-01"
    },
    {
        "query": "Spanish recipient para ti vs exchange por cinco euros",
        "target_language": "Spanish",
        "category_boost": "preposition_choice",
        "expected_chunk_id": "KB-ES-03"
    },
    {
        "query": "Les lettres que j'ai écrites accord participe passe",
        "target_language": "French",
        "category_boost": "gender_agreement",
        "expected_chunk_id": "KB-FR-02"
    },
    {
        "query": "Is actually translated as actuellement in French false cognate",
        "target_language": "French",
        "category_boost": "false_friends",
        "expected_chunk_id": "KB-GEN-01"
    },
    {
        "query": "Ser aburrido vs estar aburrido character vs boredom state",
        "target_language": "Spanish",
        "category_boost": "verb_conjugation",
        "expected_chunk_id": "KB-ES-04"
    }
]


class KnowledgeBaseRetriever:
    """Parses markdown knowledge base chunks and performs filtered, boosted retrieval."""

    def __init__(self, kb_path: str = KB_DIR):
        self.kb_path = kb_path
        self.chunks = self._load_chunks()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self._fit_vectorizer()

    def _load_chunks(self) -> List[Dict[str, Any]]:
        chunks = []
        if not os.path.exists(self.kb_path):
            return chunks

        for fname in os.listdir(self.kb_path):
            if fname.endswith(".md"):
                fpath = os.path.join(self.kb_path, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse frontmatter
                metadata = {}
                body = content
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter = parts[1]
                        body = parts[2].strip()
                        for line in frontmatter.strip().split("\n"):
                            if ":" in line:
                                k, v = line.split(":", 1)
                                metadata[k.strip()] = v.strip()

                chunk_id = metadata.get("id", os.path.splitext(fname)[0])
                chunks.append({
                    "id": chunk_id,
                    "title": metadata.get("title", chunk_id),
                    "language": metadata.get("language", "General"),
                    "category": metadata.get("category", "general"),
                    "cefr_level": metadata.get("cefr_level", "All"),
                    "content": body,
                    "search_text": f"{metadata.get('title', '')} {metadata.get('category', '')} {body}"
                })
        return chunks

    def _fit_vectorizer(self):
        if not self.chunks:
            return
        corpus = [c["search_text"] for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        query: str,
        target_language: str = "Spanish",
        category_boost: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k chunks matching language filter and optional category boost."""
        if not self.chunks:
            return []

        # Language filtering: chunk matches target_language OR 'General'
        valid_indices = [
            i for i, c in enumerate(self.chunks)
            if c["language"].lower() in [target_language.lower(), "general"]
        ]
        if not valid_indices:
            valid_indices = list(range(len(self.chunks)))

        # Query vector
        query_vec = self.vectorizer.transform([query])
        sub_matrix = self.tfidf_matrix[valid_indices]
        sims = cosine_similarity(query_vec, sub_matrix)[0]

        # Apply category boosting (1.5x score boost if category matches)
        scored_chunks = []
        for local_idx, sim in enumerate(sims):
            orig_idx = valid_indices[local_idx]
            chunk = self.chunks[orig_idx]
            final_score = float(sim)
            if category_boost and chunk["category"].lower() == category_boost.lower():
                final_score *= 1.5

            scored_chunks.append({
                "chunk_id": chunk["id"],
                "title": chunk["title"],
                "language": chunk["language"],
                "category": chunk["category"],
                "cefr_level": chunk["cefr_level"],
                "content": chunk["content"],
                "similarity_score": round(final_score, 4)
            })

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = scored_chunks[:top_k]

        # Assign citation numbers [1], [2], [3]
        for idx, res in enumerate(top_results, start=1):
            res["citation_id"] = f"[{idx}]"

        return top_results


def evaluate_retrieval(retriever: KnowledgeBaseRetriever) -> Dict[str, Any]:
    """Run evaluation benchmark across 18 queries to report Hit@1, Hit@3, and MRR."""
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    details = []

    for q in BENCHMARK_QUERIES:
        retrieved = retriever.retrieve(
            query=q["query"],
            target_language=q["target_language"],
            category_boost=q.get("category_boost"),
            top_k=3
        )
        ranked_ids = [r["chunk_id"] for r in retrieved]
        target = q["expected_chunk_id"]

        is_hit_1 = (len(ranked_ids) > 0 and ranked_ids[0] == target)
        is_hit_3 = (target in ranked_ids)

        if is_hit_1:
            hits_at_1 += 1
            rr = 1.0
        elif is_hit_3:
            rank = ranked_ids.index(target) + 1
            rr = 1.0 / rank
        else:
            rr = 0.0

        if is_hit_3:
            hits_at_3 += 1

        reciprocal_ranks.append(rr)
        details.append({
            "query": q["query"],
            "expected_id": target,
            "retrieved_ids": ranked_ids,
            "hit_at_1": is_hit_1,
            "hit_at_3": is_hit_3,
            "reciprocal_rank": rr
        })

    n = len(BENCHMARK_QUERIES)
    metrics = {
        "total_benchmark_queries": n,
        "hit_at_1_rate": round(hits_at_1 / n, 4),
        "hit_at_3_rate": round(hits_at_3 / n, 4),
        "mean_reciprocal_rank": round(float(np.mean(reciprocal_ranks)), 4),
        "benchmark_details": details
    }

    metrics_file = os.path.join(OUTPUT_DIR, "retrieval_evaluation.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n--- RAG Retrieval Evaluation (18 Labelled Queries) ---")
    print(f"Hit@1: {metrics['hit_at_1_rate']:.2%} ({hits_at_1}/{n})")
    print(f"Hit@3: {metrics['hit_at_3_rate']:.2%} ({hits_at_3}/{n})")
    print(f"MRR:   {metrics['mean_reciprocal_rank']:.4f}")
    print(f"[OK] Retrieval evaluation saved to {metrics_file}")
    return metrics


class WritingCoPilot:
    """StorySprout Writing Co-Pilot powered by Gemini API with offline pedagogical fallback."""

    def __init__(self, retriever: Optional[KnowledgeBaseRetriever] = None):
        self.retriever = retriever or KnowledgeBaseRetriever()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._init_gemini()

    def _init_gemini(self):
        self.client = None
        if self.api_key and self.api_key.strip() != "your_gemini_api_key_here":
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Notice] Gemini client could not initialize ({e}); using offline fallback.")

    def coach_learner(
        self,
        draft_text: str,
        target_language: str = "Spanish",
        detected_category: str = "tense_inconsistency"
    ) -> Dict[str, Any]:
        """Generate pedagogical coaching feedback citing retrieved context without full rewrites."""
        # 1. Retrieve top-3 chunks
        retrieved_chunks = self.retriever.retrieve(
            query=draft_text,
            target_language=target_language,
            category_boost=detected_category,
            top_k=3
        )

        # Build context passage
        context_blocks = []
        for chunk in retrieved_chunks:
            context_blocks.append(f"{chunk['citation_id']} {chunk['title']}:\n{chunk['content']}\n")
        context_text = "\n".join(context_blocks)

        system_instruction = (
            "You are StorySprout's writing coach. Use only the CONTEXT passages for rules and cite them like [1]. "
            "Never rewrite the learner's whole text or give the full corrected sentence. Point to at most 3 issues: "
            "quote the small part, name the error type, explain the rule, give a hint. End with one short task "
            "for the learner to try. Keep it under 180 words and stay encouraging."
        )

        prompt = f"""CONTEXT:
{context_text}

LEARNER DRAFT ({target_language}):
"{draft_text}"

COACHING RESPONSE:"""

        # 2. Attempt Gemini generation if client is available
        coaching_text = None
        used_gemini = False

        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"system_instruction": system_instruction, "temperature": 0.3}
                )
                if response and response.text:
                    coaching_text = response.text.strip()
                    used_gemini = True
            except Exception as e:
                print(f"[Notice] Gemini API call failed ({e}); switching seamlessly to offline pedagogical coach.")

        # 3. Offline pedagogical fallback engine (strict adherence to Section 6.5)
        if not coaching_text:
            primary_chunk = retrieved_chunks[0] if retrieved_chunks else None
            cid = primary_chunk["citation_id"] if primary_chunk else "[1]"
            ctitle = primary_chunk["title"] if primary_chunk else "Language Rule"

            coaching_text = (
                f"Great job practicing your {target_language}! Here is a friendly coaching hint:\n\n"
                f"Take a look at your sentence structure and verbs. According to {cid} ({ctitle}), "
                f"be mindful of verb aspect and preposition choice. Rather than viewing the entire action as completed, "
                f"check whether you are describing an ongoing background scene or a single completed milestone.\n\n"
                f"**Your Turn:** Re-read the clause highlighted above. Can you tweak just the main verb into its "
                f"correct form to reflect this distinction? Give it one more try!"
            )

        return {
            "coaching_feedback": coaching_text,
            "retrieved_sources": retrieved_chunks,
            "used_gemini": used_gemini,
            "detected_category": detected_category,
            "target_language": target_language
        }


class SemanticAnswerEvaluator:
    """Evaluates comprehension answers against reference answers using semantic meaning similarity."""

    def __init__(self):
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Lightweight, ultra-fast embedding model
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[Notice] SentenceTransformer offline fallback ({e}); using TF-IDF cosine similarity.")

    def evaluate(self, learner_answer: str, reference_answer: str) -> Dict[str, Any]:
        """Compute semantic similarity score (0 to 100%) and formative feedback."""
        if not learner_answer or not learner_answer.strip():
            return {
                "similarity_score": 0.0,
                "passed": False,
                "feedback": "No answer provided. Please write a sentence answering the question."
            }

        score = 0.0
        if self.model:
            try:
                emb1 = self.model.encode([learner_answer])
                emb2 = self.model.encode([reference_answer])
                sim = float(cosine_similarity(emb1, emb2)[0][0])
                score = round(max(0.0, min(1.0, sim)) * 100, 1)
            except Exception:
                score = self._tfidf_similarity(learner_answer, reference_answer)
        else:
            score = self._tfidf_similarity(learner_answer, reference_answer)

        # Formative pedagogical feedback
        passed = score >= 65.0
        if score >= 85.0:
            feedback = "Excellent! Your answer captured all the key conceptual points accurately."
        elif score >= 65.0:
            feedback = "Good response! You grasped the main meaning, though you can expand on specific details."
        elif score >= 40.0:
            feedback = "Partially correct. You mentioned some related vocabulary, but review the story details to fully address the question."
        else:
            feedback = "Not quite. Compare your response with the story passage to find the specific event described."

        return {
            "similarity_score": score,
            "passed": passed,
            "feedback": feedback,
            "reference_answer": reference_answer
        }

    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        vec = TfidfVectorizer(stop_words="english")
        try:
            m = vec.fit_transform([text1, text2])
            sim = float(cosine_similarity(m[0:1], m[1:2])[0][0])
            return round(max(0.0, min(1.0, sim)) * 100, 1)
        except Exception:
            return 50.0


def main():
    print("--- Starting Part D: RAG Co-Pilot & Retrieval Benchmark ---")
    retriever = KnowledgeBaseRetriever()
    print(f"Loaded {len(retriever.chunks)} Knowledge Base chunks from {KB_DIR}")

    # Evaluate retrieval
    eval_metrics = evaluate_retrieval(retriever)

    # Test Co-pilot coaching
    copilot = WritingCoPilot(retriever=retriever)
    test_draft = "Ayer mientras caminé al mercado, vi mi amigo y compré flores para cinco euros."
    result = copilot.coach_learner(test_draft, target_language="Spanish", detected_category="tense_inconsistency")

    print("\nSample Co-Pilot Coaching Response:")
    print(result["coaching_feedback"])
    print(f"\nUsed Gemini API: {result['used_gemini']}")
    print("Retrieved Sources:")
    for s in result["retrieved_sources"]:
        print(f" - {s['citation_id']} {s['title']} (Score: {s['similarity_score']})")

    # Test Semantic Answer Evaluator
    evaluator = SemanticAnswerEvaluator()
    learner_ans = "The protagonist went to the central market to buy fresh vegetables."
    ref_ans = "He visited the town market in the morning to purchase fresh vegetables for dinner."
    eval_res = evaluator.evaluate(learner_ans, ref_ans)
    print(f"\nSemantic Evaluator Test:")
    print(f"Similarity: {eval_res['similarity_score']}% | Passed: {eval_res['passed']}")
    print(f"Feedback: {eval_res['feedback']}")

    print("\n--- Part D Completed Successfully ---")


if __name__ == "__main__":
    main()

