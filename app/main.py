from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from snowflake.snowpark import Session
from snowflake.core import Root
from dotenv import load_dotenv
import os

# Load env vars
load_dotenv()

# Init OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o-mini"  # or gpt-4o-mini

# FastAPI app
app = FastAPI()

# Snowflake setup
def create_snowflake_session():
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "test_warehouse"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "CORTEX_SEARCH_IPAM"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
    }
    return Session.builder.configs(connection_parameters).create()

session = create_snowflake_session()
root = Root(session)

# Cortex Search service
CORTEX_SERVICE_NAME = "BOOKS_DATASET_SERVICE"
CORTEX_SEARCH_COLUMNS = ["CHUNK", "TITLE", "AUTHORS", "DESCRIPTION","CATEGORY","Publisher","PRICE"]

# Request body
class ChatRequest(BaseModel):
    question: str
    use_chat_history: bool = True
    history: list = []

# Query Cortex
def query_cortex_search_service(query: str, limit: int = 5):
    try:
        cortex_service = (
            root
            .databases["CORTEX_SEARCH_IPAM"]
            .schemas["PUBLIC"]
            .cortex_search_services[CORTEX_SERVICE_NAME]
        )

        results = cortex_service.search(
            query=query,
            columns=CORTEX_SEARCH_COLUMNS,
            limit=limit
        ).results

        if not results:
            return ""

        context_docs = ""
        for i, row in enumerate(results):
            context_docs += f"\nDocument {i+1}:\n"
            for col in CORTEX_SEARCH_COLUMNS:
                context_docs += f"{col.title()}: {row.get(col, '')}\n"

        return context_docs.strip()

    except Exception as e:
        return f"❌ Error querying service: {str(e)}"

# Prompt builder
def generate_prompt(question: str, context: str, history: list):
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    prompt = f"""
[INST]
You are a helpful assistant with access to book information.

<chat_history>
{history_text}
</chat_history>
<context>
{context}
</context>
<question>
{question}
</question>
[/INST]
"""
    return prompt

# Get OpenAI completion
def get_completion(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()

# API route
@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required.")

    # Search with history context if enabled
    search_query = req.question
    if req.use_chat_history and req.history:
        history_text = " ".join([m['content'] for m in req.history])
        search_query = f"{history_text} {req.question}"

    context = query_cortex_search_service(search_query)

    if not context or context.lower().startswith("❌ error"):
        return {
            "response": "❗ Sorry, I couldn't find anything relevant in the data."
        }

    final_prompt = generate_prompt(req.question, context, req.history)
    answer = get_completion(final_prompt)

    return {"response": answer}
