"""
src/api/schemas.py
------------------
Pydantic models for API request / response validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class VitalReading(BaseModel):
    timestamp: datetime
    hr:   Optional[float] = Field(None, ge=25,  le=220,  description="Heart rate (bpm)")
    sbp:  Optional[float] = Field(None, ge=60,  le=220,  description="Systolic BP (mmHg)")
    dbp:  Optional[float] = Field(None, ge=30,  le=140,  description="Diastolic BP (mmHg)")
    spo2: Optional[float] = Field(None, ge=70,  le=100,  description="SpO₂ (%)")
    rr:   Optional[float] = Field(None, ge=4,   le=50,   description="Respiratory rate (breaths/min)")
    temp: Optional[float] = Field(None, ge=34,  le=42,   description="Temperature (°C)")


class PredictRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    vitals_window: list[VitalReading] = Field(
        ..., min_length=3, max_length=500,
        description="Sequence of vital sign readings (up to 6 hours)",
    )

    @field_validator("vitals_window")
    @classmethod
    def sorted_by_time(cls, readings: list[VitalReading]) -> list[VitalReading]:
        return sorted(readings, key=lambda r: r.timestamp)

    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "P-00042",
                "vitals_window": [
                    {"timestamp": "2024-01-15T08:00:00Z", "hr": 88, "sbp": 122, "dbp": 78, "spo2": 97, "rr": 16, "temp": 37.1},
                    {"timestamp": "2024-01-15T08:30:00Z", "hr": 91, "sbp": 118, "dbp": 76, "spo2": 96, "rr": 17, "temp": 37.2},
                    {"timestamp": "2024-01-15T09:00:00Z", "hr": 95, "sbp": 115, "dbp": 74, "spo2": 95, "rr": 18, "temp": 37.3},
                ],
            }
        }
    }


class FeatureImportance(BaseModel):
    name: str
    shap_value: float


class PredictResponse(BaseModel):
    patient_id: str
    deterioration_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH")
    predicted_at: datetime
    model_version: str
    top_features: list[FeatureImportance] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float