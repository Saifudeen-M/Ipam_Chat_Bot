from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from snowflake.snowpark import Session
from snowflake.core import Root
from dotenv import load_dotenv
import os
import time
import logging

# Load environment variables
load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o-mini"

# FastAPI app
app = FastAPI()

# Enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware: Log requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logging.info(f"{request.method} {request.url} completed in {duration:.2f}s with status {response.status_code}")
    return response

# Snowflake session setup
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

# Cortex Search details
CORTEX_SERVICE_NAME = "BOOKS_DATASET_SERVICE"
CORTEX_SEARCH_COLUMNS = ["CHUNK", "TITLE", "AUTHORS", "DESCRIPTION", "CATEGORY", "Publisher", "PRICE"]

# Request model
class ChatRequest(BaseModel):
    question: str
    use_chat_history: bool = True
    history: list = []

# Query Cortex service
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

# Build prompt
def generate_prompt(question: str, context: str, history: list):
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    prompt = f"""
        [INST]
        You are BookBot, a helpful and conversational AI assistant.

        You can freely engage in general conversation like greetings or casual questions (e.g., "hi", "how are you", "what can you do?").

        For **book-related questions** (such as about title, author, category, price, or description), follow these strict rules:

        1. Use the <context> provided below as the **primary and only data source**.
        2. You can explain, rephrase, or elaborate using natural language — **but all factual details must come from the context**.
        3. If the context is empty or not relevant to the question, respond with:
        4. If someone asks "how were you created?", "how were you born?", or "who made you?", respond with:
        - My name is **Genz**.
        - I was created by **Saifudeen**, who designed me to be a smart, friendly assistant for book lovers.
        - Here's how I work:
            🧑‍💻 You type a message or question.
            💬 It’s sent through a chat interface on the frontend.
            🚀 The message is passed to a backend API built using FastAPI.
            🔍 The API queries a real-time book dataset using **Snowflake Cortex Search**.
            🤖 Then, GPT-4o-mini (by OpenAI) generates a natural, friendly response using that data.
            📲 Finally, the answer is shown back to you in the chat.
        - I combine real-time book data with powerful AI language skills to help you quickly and clearly.
        "I'm sorry, I couldn't find relevant information in the book dataset."
        ---

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

# Get OpenAI response
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

# POST endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required.")

    # Combine chat history with question for better search context
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