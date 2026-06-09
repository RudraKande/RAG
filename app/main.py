from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from rag import ask_question

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "RAG server running"
    }


@app.post("/ask", response_class=PlainTextResponse)
def ask(question: str):

    answer = ask_question(question)

    return answer
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)