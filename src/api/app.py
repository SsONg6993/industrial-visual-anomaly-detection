from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from PIL import Image, UnidentifiedImageError

from src.api.inference_service import (
    InferenceService,
)

from src.api.schemas import (
    HealthResponse,
    PredictionResponse,
)


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# Global service instance
# ============================================================

service: InferenceService | None = None


# ============================================================
# Application lifespan
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    global service

    print(
        "Loading anomaly detection service..."
    )

    service = InferenceService(
        PROJECT_ROOT
    )

    print(
        f"Service ready | "
        f"device={service.device_name} | "
        f"coreset={service.coreset_size} | "
        f"threshold={service.threshold:.6f}"
    )

    yield

    # Optional cleanup
    service = None


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title=(
        "Industrial Visual "
        "Anomaly Detection API"
    ),

    description=(
        "Industrial anomaly detection "
        "using ResNet18 features and "
        "a normal-reference memory bank."
    ),

    version="1.0.0",

    lifespan=lifespan,
)


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "Industrial Visual "
            "Anomaly Detection API",

        "docs":
            "/docs",

        "health":
            "/health",

        "predict":
            "/predict",
    }


# ============================================================
# GET /health
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    if (
        service is None
        or
        not service.is_ready
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Inference service "
                "is not ready."
            )
        )


    return HealthResponse(
        status="ok",

        device=(
            service.device_name
        ),

        model_loaded=(
            service.is_ready
        ),

        coreset_size=(
            service.coreset_size
        ),
    )


# ============================================================
# POST /predict
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------
    # Service availability
    # -----------------------------------------

    if (
        service is None
        or
        not service.is_ready
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Inference service "
                "is not ready."
            )
        )


    # -----------------------------------------
    # Content type validation
    # -----------------------------------------

    if (
        file.content_type is None
        or
        not file.content_type.startswith(
            "image/"
        )
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file must "
                "be an image."
            )
        )


    # -----------------------------------------
    # Read upload
    # -----------------------------------------

    try:

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded image "
                    "is empty."
                )
            )


        image = Image.open(
            BytesIO(
                contents
            )
        )

        image.load()

        image = image.convert(
            "RGB"
        )


    except (
        UnidentifiedImageError,
        OSError
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode "
                "uploaded image."
            )
        )


    # -----------------------------------------
    # Inference
    # -----------------------------------------

    try:

        result = service.predict(
            image
        )


    except HTTPException:

        raise


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Inference failed: "
                f"{str(exc)}"
            )
        )


    return PredictionResponse(
        prediction=(
            result[
                "prediction"
            ]
        ),

        anomaly_score=(
            result[
                "anomaly_score"
            ]
        ),

        threshold=(
            result[
                "threshold"
            ]
        ),

        latency_ms=(
            result[
                "latency_ms"
            ]
        ),
    )