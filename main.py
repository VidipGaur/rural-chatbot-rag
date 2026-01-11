from collections import defaultdict

conversation_store = defaultdict(list)
from app.rules import is_emergency, basic_severity_hint
from app.languages import LANG_CONFIG
from app.rag import retrieve_context
from app.llm import generate_llm_reply


from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def translate_to_english(text: str, language: str) -> str:
    if language == "en":
        return text

    if language == "kn":
        tokenizer.src_lang = "kan_Knda"
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=50,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text

def translate_to_kannada(text_en: str) -> str:
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text_en, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("kan_Knda"),
        max_length=80,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

app = FastAPI()

class ChatRequest(BaseModel):
    text: str
    language: str  # "en" or "kn"
    session_id: str

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    if src_lang == tgt_lang:
        return text

    tokenizer.src_lang = LANG_CONFIG[src_lang]["nllb_code"]
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            LANG_CONFIG[tgt_lang]["nllb_code"]
        ),
        max_length=100,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/chat")
def chat(req: ChatRequest):
    user_lang = req.language
    session_id = req.session_id

    text_en = translate(req.text, user_lang, "en")

    # 🚨 Emergency shortcut (no LLM)
    if is_emergency(text_en):
        warning_en = "This may be a medical emergency. Please seek immediate medical care."
        reply = translate(warning_en, "en", user_lang)
        return {"reply": reply}

    # optional severity hint (for prompt conditioning later)
    severity = basic_severity_hint(text_en)

    conversation_store[session_id].append(
        {"role": "user", "content": text_en}
    )

    history = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in conversation_store[session_id]
    )

    context = retrieve_context(text_en)
    reply_en = generate_llm_reply(context, history)

    conversation_store[session_id].append(
        {"role": "assistant", "content": reply_en}
    )

    reply_user_lang = translate(reply_en, "en", user_lang)
    return {"reply": reply_user_lang}

