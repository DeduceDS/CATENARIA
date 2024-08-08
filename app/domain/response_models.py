# /app/domain/response_models.py
from pydantic import BaseModel


class VanoPrediction(BaseModel):
    ID_VANO: str
    CONDUCTORES_CORREGIDOS: dict
    PARAMETROS_a_h_k: dict
    FLAG: str
    NUM_CONDUCTORES: int
    NUM_CONDUCTORES_FIABLE: bool
    CONFIG_CONDUCTORES: int
    COMPLETITUD: str
    RECONSTRUCCION: str
    PORCENTAJE_HUECOS: float
    # CONDUCTORES_CORREGIDOS_PARAMETROS: dict
    ERROR_POLILINEA: float
    ERROR_CATENARIA: float
