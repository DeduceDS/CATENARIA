from app.domain.models import *


class VanoPrediction(BaseModel):
    PARAMETROS_a_h_k: dict
    RECONSTRUCION: str
    CONDUCTORES_CORREGIDOS: dict
    CONDUCTORES_CORREGIDOS_PARAMETROS: dict
    FLAG: str
    NUM_CONDUCTORES: int
    NUM_CONDUCTORES_FIABLES: bool
    CONFIG_CONDUCTORES: int
    COMPLETITUD: str
    ERROR_POLILINEA: float
    ERROR_CATENARIA: float


class VanoPredictionResponseSchema(BaseModel):
    ID_VANO: str
    PREDICCION: VanoPrediction
