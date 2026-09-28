import os
import uuid
import asyncio
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from qdrant_client import models, QdrantClient

load_dotenv()

llm = AsyncOpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
title_client = AsyncOpenAI(base_url=os.getenv("TITLE_LEGACY_BASE_URL"), api_key="dummy")
client = QdrantClient(url=os.getenv("QDRANT_URL"))

MODEL_TITLE = "SupraLabs/Supra-Title-350M-exp-GGUF"


async def appeler_llm(messages: list) -> dict:
    response = await llm.chat.completions.create(
        model="gemma-4-31b",
        messages=messages,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    message = response.choices[0].message
    return {"reasoning": message.reasoning or "", "content": message.content}


async def generer_titre(question: str) -> str:
    response = await title_client.chat.completions.create(
        model=MODEL_TITLE,
        messages=[
            {
                "role": "system",
                "content": "Generate a title in 3 to 5 words maximum. Never exceed 5 words. Always respond in English only."
            },
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=20,
        extra_body={"repetition_penalty": 1.05},
    )
    return response.choices[0].message.content.strip()



async def generer_label_thinking(phrase: str) -> str:
    response = await title_client.chat.completions.create(
        model=MODEL_TITLE,
        messages=[
            {
                "role": "system",
                "content": "Generate a label in 4 to 5 words maximum. Never exceed 5 words. Always respond in English."
            },
            {
                "role": "user",
                "content": (
                    f"This is an AI's internal reasoning fragment. "
                    f"Generate ONE ultra-short label (4-5 words) describing what the AI is doing RIGHT NOW. "
                    f"If it's analyzing a concept → 'Analyzing [concept]'. "
                    f"If it's planning structure → 'Planning response structure'. "
                    f"If it's comparing options → 'Comparing [option A] vs [option B]'. "
                    f"If it's checking facts → 'Checking [specific fact]'. "
                    f"If it's thinking about the user → 'Understanding user intent'. "
                    f"Start with a present tense verb. Be specific to the content. "
                    f"Always respond in English regardless of the fragment's language. "
                    f"Generate ONE label of EXACTLY 3 to 5 words. "
                    f"Never generate less than 3 words. Never generate more than 5 words. "
                    f"Never use generic phrases like 'Understanding User Intent'. "
                    f"Be specific to the content of the fragment. "
                    f"Fragment: {phrase[:300]}"
                ),
            }
        ],
        temperature=0.2,
        max_tokens=20,
        extra_body={"repetition_penalty": 1.05},
    )
    
    return response.choices[0].message.content.strip()


def sauvegarder_conversation(titre: str, messages: list) -> None:
    if not client.collection_exists(collection_name="conversations"):
        client.create_collection(
            collection_name="conversations",
            vectors_config=models.VectorParams(
                size=1,
                distance=models.Distance.COSINE,
            ),
        )
    client.upsert(
        collection_name="conversations",
        points=[
            models.PointStruct(
                id=uuid.uuid4(),
                vector=[0.0],
                payload={
                    "titre": titre,
                    "messages": messages,
                    "created_at": datetime.now().isoformat(),
                },
            )
        ],
    )


def charger_historique() -> list:
    if not client.collection_exists(collection_name="conversations"):
        return []
    results = client.scroll(
        collection_name="conversations", limit=50, with_payload=True, with_vectors=False
    )
    points = results[0]
    points.sort(key=lambda p: p.payload.get("created_at", ""), reverse=True)
    return points


async def main():
    question = await asyncio.to_thread(input, "Vous : ")
    messages = [{"role": "user", "content": question}]

    result = await appeler_llm(messages)
    messages.append({"role": "assistant", "content": result["content"]})
    print(f"\nAssistant : {result['content']}")

    titre = await generer_titre(question)
    sauvegarder_conversation(titre, messages)
    print(f"\nTitre : {titre}")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())

