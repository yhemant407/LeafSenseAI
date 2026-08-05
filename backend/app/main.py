from fastapi import FastAPI

from app.api.router import router

app = FastAPI(
    title="LeafSense AI API",
    version="1.0.0",
    description="AI-powered plant disease detection service."
)


app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "LeafSense AI API is running!"
    }