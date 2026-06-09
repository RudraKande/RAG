from logger import logger
from vectorstore import db
from llm import llm


def ask_question(question):

    logger.info("="*80)
    logger.info(f"Question : {question}")

    docs = db.similarity_search(
        question,
        k=3
    )

    logger.info(f"Retrieved {len(docs)} chunks")

    for i, doc in enumerate(docs):

        logger.info(f"Chunk {i+1}")
        logger.info(doc.page_content[:500])

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    logger.info("Sending context to LLM...")

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context below.
Format answers in a natural, easy-to-read manner.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    logger.info("LLM generated response")
    logger.info(response.content)

    logger.info("="*80)

    return response.content