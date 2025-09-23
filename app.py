import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = "./gemma-2-2b-it"
lora_path = "./gemma-lora/checkpoint-2600"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class Query(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
def ask(query: Query):
    if not query.question.strip():
        return {"answer": "Please enter a question."}

    prompt = f"<start_of_turn>user\n{query.question}<end_of_turn>\n<start_of_turn>model\n"

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if prompt in raw:
        candidate = raw.split(prompt, 1)[1]
    else:
        m = re.search(r"<start_of_turn>\s*model\s*", raw, flags=re.IGNORECASE)
        if m:
            candidate = raw[m.end():]
        else:
            m2 = re.search(r"(^|\n)\s*model[:\-\s]*", raw, flags=re.IGNORECASE)
            if m2:
                candidate = raw[m2.end():]
            else:
                candidate = raw

    cut = re.search(r"<end_of_turn>|<start_of_turn>\s*user", candidate, flags=re.IGNORECASE)
    if cut:
        candidate = candidate[:cut.start()]

    candidate = re.sub(r"<.*?>", "", candidate)
    candidate = re.sub(r'^\s*model[:\-\s]*', '', candidate, flags=re.IGNORECASE)
    answer = candidate.strip()

    return {"answer": answer}
