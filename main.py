from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from similarity import calculate_similarity
"""THIS WOULD BE THE SIMPLEST FASTAPI CODE TO SERVE ALL THE REQUIREMENTS OF THE PROJECT AS ASKED """

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "score": None})

def get_score(text1, text2):
    return calculate_similarity(text1, text2)

@app.post("/")
async def calculate_similarity_form(request: Request, text1: str = Form(...), text2: str = Form(...)):
    score = get_score(text1, text2)
    return templates.TemplateResponse("index.html", {"request": request, "score": score, "text1": text1, "text2": text2})