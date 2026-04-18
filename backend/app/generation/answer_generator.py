import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self, api_key: str = None):

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "❌ GROQ API key not found. Set it in .env or pass it explicitly."
            )

        self.client = Groq(api_key=self.api_key)

    # ✅ NORMAL (non-streaming)
    def generate(self, query: str, retrieved_chunks: list[str]) -> str:

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are an enterprise knowledge assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not present in the context, say:
  "I don't have enough information to answer this question."
- Be concise and precise.

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logging.error("❌ Error generating answer: %s", str(e))
            return "Error generating response. Please try again."

    # 🔥 STREAMING VERSION (NEW)
    def stream_generate(self, query: str, retrieved_chunks: list[str]):

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are an enterprise knowledge assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not present in the context, say:
  "I don't have enough information to answer this question."
- Be concise and precise.

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            stream = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stream=True  # 🔥 KEY
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logging.error("❌ Streaming error: %s", str(e))
            yield "Error generating response."

    # ─── Query Rewriting (Task 5) ────────────────────────────────────────────

    def rewrite_query(self, query: str) -> str:
        """
        Expand a short or vague query into a more specific retrieval query.

        Example:
            "leave policy"
            → "employee annual leave policy entitlement company rules"

        Returns the original query unchanged on any error or empty output.
        Uses a single short LLM call (max_tokens=64) to keep latency low.
        """
        prompt = (
            "Rewrite the following search query to be more specific and descriptive "
            "for retrieving relevant enterprise documents. "
            "Output only the rewritten query — no explanation, no quotes.\n\n"
            f"Original query: {query}\n"
            "Rewritten query:"
        )
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64,
            )
            rewritten = response.choices[0].message.content.strip()
            # Sanity checks: non-empty, not excessively long, not a refusal
            if rewritten and len(rewritten) <= 300 and len(rewritten) >= len(query) // 2:
                return rewritten
        except Exception as exc:
            logger.warning("Query rewriting failed, using original query: %s", exc)
        return query