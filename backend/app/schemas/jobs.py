"""Pydantic contracts for the local bakery image-processing API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


InferenceMode = Literal["AUTO", "YOLO", "FOUNDATION", "COMPARE"]
DocumentType = Literal["PURCHASE_RECEIPT", "MANUFACTURING"]


JobState = Literal[
    "QUEUED",
    "PROCESSING",
    "AWAITING_CONFIRMATION",
    "NEEDS_RETAKE",
    "CONFIRMING",
    "COMPLETED",
    "ERROR",
]


class ProductSummary(BaseModel):
    product_code: str
    product_name: str
    purchase_price: float = 0
    quantity: int = Field(ge=1)


class ImageSummary(BaseModel):
    image_name: str
    status: Literal["SUCCESS", "ERROR"]
    total_detections: int = Field(default=0, ge=0)
    avg_confidence: float = Field(default=0, ge=0, le=1)
    inference_ms: float = Field(default=0, ge=0)
    annotated_filename: str | None = None
    annotated_url: str | None = None
    error: str = ""


class JobAccepted(BaseModel):
    job_id: str
    status: JobState
    total_images: int = Field(ge=1)
    status_url: str
    message: str


class UploadFileRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=100)
    size_bytes: int = Field(gt=0)


class PresignUploadsRequest(BaseModel):
    job_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    files: list[UploadFileRequest] = Field(min_length=1)
    inference_mode: InferenceMode = "AUTO"


class PresignedUpload(BaseModel):
    filename: str
    content_type: str
    size_bytes: int = Field(gt=0)
    object_key: str
    upload_url: str
    method: Literal["PUT"] = "PUT"
    headers: dict[str, str]


class PresignUploadsResponse(BaseModel):
    job_id: str
    expires_in: int = Field(gt=0)
    uploads: list[PresignedUpload]
    submit_url: str


class R2JobFile(BaseModel):
    object_key: str = Field(min_length=1, max_length=1024)


class CreateR2JobRequest(BaseModel):
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    files: list[R2JobFile] = Field(min_length=1)
    inference_mode: InferenceMode = "AUTO"


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobState
    created_at: str
    updated_at: str
    total_images: int = Field(ge=1)
    processed_images: int = Field(default=0, ge=0)
    error: str = ""
    message: str = ""
    confirmation_error: str = ""
    decision: dict[str, Any] | None = None
    confirmed_product: dict[str, Any] | None = None
    product_count: int = Field(default=0, ge=0)
    total_quantity: int = Field(default=0, ge=0)
    products: list[ProductSummary] = Field(default_factory=list)
    images: list[ImageSummary] = Field(default_factory=list)
    excel_filename: str | None = None
    excel_url: str | None = None
    r2_objects: list[dict[str, Any]] = Field(default_factory=list)
    kiotviet: dict[str, Any] | None = None
    document_type: DocumentType | None = None
    inference_mode: InferenceMode = "AUTO"
    pseudo_label: dict[str, Any] | None = None


class KiotVietSubmitRequest(BaseModel):
    confirm: bool = False


class BakeryHealthResponse(BaseModel):
    ready: bool
    template_ready: bool
    model: dict[str, Any] | None = None
    r2_configured: bool = False
    kiotviet_configured: bool = False
    manufacturing_configured: bool = False
    kiotviet_auto_create_draft: bool = False
    max_images_per_job: int = Field(default=50, ge=1)
    max_image_size_mb: float = Field(default=50, gt=0)
    max_job_upload_size_mb: float = Field(default=200, gt=0)
    allowed_image_extensions: list[str] = Field(default_factory=list)
    pseudo_label: dict[str, Any] | None = None
    error: str = ""
