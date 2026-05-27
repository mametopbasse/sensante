import os
import joblib
<<<<<<< HEAD
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import numpy as np
from pydantic import BaseModel, Field

# =========================================================
# Configuration de l'environnement & Client Groq
# =========================================================
load_dotenv()

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé avec succès.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvée. La route /explain sera désactivée.")

# =========================================================
# Schémas Pydantic
# =========================================================


class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic prédit par le modèle")
    probabilite: float = Field(..., description="Probabilité du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en français")
    modele_llm: str = Field(
        default="llama-3.1-8b-instant", description="Modèle LLM utilisé"
    )


class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str


# =========================================================
# Application FastAPI
# =========================================================
=======
import numpy as np

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from groq import Groq

from pydantic import BaseModel, Field

# =========================================================
# Application FastAPI
# =========================================================

>>>>>>> 2ec48ae (Initial clean commit)
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0",
)

<<<<<<< HEAD
=======
# =========================================================
# Frontend statique
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)

@app.get("/")
def serve_frontend():
    """Servir la page d'accueil."""
    return FileResponse("frontend/index.html")

# =========================================================
# Middleware CORS
# =========================================================

>>>>>>> 2ec48ae (Initial clean commit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
<<<<<<< HEAD
# Chargement du modèle Machine Learning
# =========================================================
=======
# Configuration environnement + Groq
# =========================================================

load_dotenv()

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé avec succès.")
else:
    print(
        "ATTENTION : GROQ_API_KEY non trouvée. "
        "La route /explain sera désactivée."
    )

# =========================================================
# Chargement du modèle ML
# =========================================================

>>>>>>> 2ec48ae (Initial clean commit)
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

<<<<<<< HEAD
# Prompt Système pour le LLM
SYSTEM_PROMPT = """Tu es un assistant médical sénégalais.
Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple,
comme un médecin parlerait à son patient.
Sois rassurant mais recommande toujours
une consultation médicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-même.
Tu expliques uniquement le diagnostic fourni."""


# =========================================================
# Routes de l'API
# =========================================================


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}
=======
# =========================================================
# Prompt système
# =========================================================

SYSTEM_PROMPT = """
Tu es un assistant médical sénégalais.

Tu reçois un diagnostic et des données patient.

Explique le résultat en français simple,
comme un médecin parlerait à son patient.

Sois rassurant mais recommande toujours
une consultation médicale.

Maximum 3 phrases.

Ne fais JAMAIS de diagnostic toi-même.

Tu expliques uniquement le diagnostic fourni.

Si la confiance est faible,
précise clairement que le résultat est incertain.
"""

# =========================================================
# Schémas Pydantic
# =========================================================

class ExplainInput(BaseModel):
    diagnostic: str = Field(
        ...,
        description="Diagnostic prédit par le modèle"
    )

    probabilite: float = Field(
        ...,
        description="Probabilité du diagnostic"
    )

    confiance: str = Field(
        ...,
        description="Niveau de confiance"
    )

    age: int = Field(...)

    sexe: str = Field(...)

    temperature: float = Field(...)

    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(
        ...,
        description="Explication en français"
    )

    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modèle LLM utilisé"
    )


class PatientInput(BaseModel):

    age: int = Field(..., ge=0, le=120)

    sexe: str = Field(...)

    temperature: float = Field(..., ge=35.0, le=42.0)

    tension_sys: int = Field(..., ge=60, le=250)

    toux: bool = Field(...)

    fatigue: bool = Field(...)

    maux_tete: bool = Field(...)

    region: str = Field(...)


class DiagnosticOutput(BaseModel):

    diagnostic: str

    probabilite: float

    confiance: str

    message: str

# =========================================================
# Routes API
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "message": "SenSante API is running"
    }
>>>>>>> 2ec48ae (Initial clean commit)


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
<<<<<<< HEAD
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
=======

    # Encodage sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]

    except ValueError:

>>>>>>> 2ec48ae (Initial clean commit)
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}",
        )

<<<<<<< HEAD
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
=======
    # Encodage région
    try:
        region_enc = le_region.transform([patient.region])[0]

    except ValueError:

>>>>>>> 2ec48ae (Initial clean commit)
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Région inconnue : {patient.region}",
        )

<<<<<<< HEAD
    features = np.array(
        [
            [
                patient.age,
                sexe_enc,
                patient.temperature,
                patient.tension_sys,
                int(patient.toux),
                int(patient.fatigue),
                int(patient.maux_tete),
                region_enc,
            ]
        ]
    )

    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())

    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

=======
    # Features modèle
    features = np.array([
        [
            patient.age,
            sexe_enc,
            patient.temperature,
            patient.tension_sys,
            int(patient.toux),
            int(patient.fatigue),
            int(patient.maux_tete),
            region_enc,
        ]
    ])

    # Prédiction
    diagnostic = model.predict(features)[0]

    proba_max = float(
        model.predict_proba(features)[0].max()
    )

    # Niveau confiance
    if proba_max >= 0.7:
        confiance = "haute"

    elif proba_max >= 0.4:
        confiance = "moyenne"

    else:
        confiance = "faible"

    # Messages
>>>>>>> 2ec48ae (Initial clean commit)
    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoïde. Consultation nécessaire.",
        "sain": "Pas de pathologie détectée.",
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
<<<<<<< HEAD
        message=messages.get(diagnostic, "Consultez un médecin."),
=======
        message=messages.get(
            diagnostic,
            "Consultez un médecin."
        ),
>>>>>>> 2ec48ae (Initial clean commit)
    )


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
<<<<<<< HEAD
    """Expliquer un diagnostic en français avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication=(
                "Service d'explication indisponible. Clé API non configurée."
=======
    """
    Expliquer un diagnostic en français avec un LLM.
    """

    # Dégradation gracieuse
    if not groq_client:

        return ExplainOutput(
            explication=(
                "Service d'explication indisponible. "
                "Clé API non configurée."
>>>>>>> 2ec48ae (Initial clean commit)
            ),
            modele_llm="aucun",
        )

<<<<<<< HEAD
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, région {data.region}\n"
        f"Température : {data.temperature} C\n"
        f"Diagnostic du modèle : {data.diagnostic} (probabilité {data.probabilite:.0%})\n"
=======
    # Prompt utilisateur
    user_prompt = (
        f"Patient : {data.sexe}, "
        f"{data.age} ans, région {data.region}\n"

        f"Température : {data.temperature} C\n"

        f"Diagnostic du modèle : "
        f"{data.diagnostic} "
        f"(probabilité {data.probabilite:.0%}, "
        f"confiance {data.confiance})\n"

>>>>>>> 2ec48ae (Initial clean commit)
        f"Explique ce résultat au patient."
    )

    try:
<<<<<<< HEAD
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        explication = response.choices[0].message.content
        modele_utilise = "llama-3.1-8b-instant"
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
        modele_utilise = "erreur"

    return ExplainOutput(explication=explication, modele_llm=modele_utilise)
=======

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],

            max_tokens=200,

            temperature=0.3,
        )

        explication = response.choices[0].message.content

        modele_utilise = "llama-3.1-8b-instant"

    except Exception as e:

        explication = (
            f"Erreur lors de l'appel au LLM : {str(e)}"
        )

        modele_utilise = "erreur"

    return ExplainOutput(
        explication=explication,
        modele_llm=modele_utilise
    )
>>>>>>> 2ec48ae (Initial clean commit)
