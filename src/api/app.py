from fastapi import FastAPI
from pydantic import BaseModel

from src.core.model_registry import model_registry

app = FastAPI(title="EvalOps-Lite", version="1.0.0")

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    # Redirect root to Swagger UI so recruiters instantly see the product
    return RedirectResponse(url="/docs")

class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "EvalOps-Lite"}


@app.get("/models")
def models():
    return model_registry.info()


@app.post("/genai/predict")
def genai_predict(req: PredictRequest):
    return model_registry.genai.predict(req.text)
