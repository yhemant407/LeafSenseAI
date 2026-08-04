from fastapi import FastAPI

app = FastAPI(
    title="Plant Disease Classifier API",
    version="1.0.0",
    description="AI-powered API for detecting plant diseases from leaf images."
)


@app.get("/")
def root():
    return {
        "message": "Plant Disease Classifier API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }