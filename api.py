import os
from pydoc import text
from pymongo import MongoClient
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from logic import build_prompt
from logic import get_embedding, find_similar, build_prompt, collection, client, MODEL



load_dotenv()
# tạo app
app = FastAPI()



# dữ liệu nhận vào
class Data(BaseModel):
    text: str

# test xem API chạy chưa
@app.get("/")
def home():
    return {"message": "API dang chay"}

# API chính
@app.post("/summarize")
def summarize(data: Data):
    text = data.text.strip()
    if not text:
        return {"summary": "Noi dung rong"}

    new_embedding = get_embedding(text)
    doc, score = find_similar(new_embedding)

    if doc and score > 0.9:
        return {
            "summary": doc["summary"],
            "mode": "reuse",
            "similarity": score
        }

    # 🔥 PROMPT CHUẨN THEO FILE MẪU
    prompt = build_prompt(text)

    response = client.responses.create(
            model=MODEL,
            input=prompt
        )

    summary = response.output_text.strip()
    # collection.insert_one({
    #     "input_text": text,
    #     "summary": summary,
    #     "embedding": new_embedding,
    #     "created_at": datetime.now(timezone.utc)
    # })

    return {
        "summary": summary,
        "mode": "new"
    }
