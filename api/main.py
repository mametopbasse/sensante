from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware   # ← import manquant
from pydantic import BaseModel, Field
import joblib
import numpy as np

# ... Schemas Pydantic (PatientInput, DiagnosticOutput) ...

# Création de app
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)

# Middleware CORS — après la création de app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... reste du code (chargement modèle, routes) ...