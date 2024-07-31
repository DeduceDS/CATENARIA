from app.domain.models import *


class VanoPredictionResponse(BaseModel):
    ID_VANO: str
    PARAMETROS: dict
    RECONSTRUCION: str
    CONDUCTORES_CORREGIDOS: dict
    CONDUCTORES_CORREGIDOS_PARAMETROS: dict
    FLAG: str
    NUM_CONDUCTORES: int
    CONFIG_CONDUCTORES: int
    COMPLETITUD: str
    ERROR_POLILINEA: float
    ERROR_CATENARIA: float
