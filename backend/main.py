# File: backend/main.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# TODO: Implement API routes and main backend logic

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)