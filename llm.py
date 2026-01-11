from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_llm_reply(context: str, user_query: str) -> str:
    prompt = f"""
You are a healthcare triage assistant.

Rules:
- Use ONLY the provided context.
- Do NOT diagnose diseases.
- Do NOT prescribe medication.
- Do NOT give severity labels unless explicitly present in context.
- If symptoms are incomplete or duration/severity is missing,
  ASK FOLLOW-UP QUESTIONS before giving advice.
- Ask at most 2 short follow-up questions.
- Be calm and reassuring.
  Do NOT repeat questions already asked.
- Treat the LAST USER message as an answer to your previous question.
- If enough information is available, give guidance.
- Otherwise, ask ONE new follow-up question.
You are allowed to ask questions like:
- How long have you had these symptoms?
- Do you have a high fever or mild fever?
- Are there any additional symptoms?

User message:
{user_query}

Your response:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=250,
        temperature=0.3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
