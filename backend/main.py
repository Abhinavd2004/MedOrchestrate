from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

app = FastAPI(title="MedOrchestrate")

# Vite's default dev server port is 5173 (confirmed: frontend/package.json
# has no vite.config port override, so it runs on Vite's default);
# 3000 is also allowed since that's the common CRA/Next.js default, in
# case the frontend setup changes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
