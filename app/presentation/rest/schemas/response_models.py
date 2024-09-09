# /app/domain/response_models.py
from pydantic import BaseModel


# TODO use fileds for schema descriptions
class PuntuacionApriori(BaseModel):
    P_HUECO: list[float]
    DIFF2D: list[float]
    NOTA: float


class VanoPrediction(BaseModel):
    ID_VANO: str
    CONDUCTORES_CORREGIDOS: dict
    PARAMETROS_a_h_k: dict
    FLAG: str = None
    NUM_CONDUCTORES: int
    NUM_CONDUCTORES_FIABLE: bool = False
    CONFIG_CONDUCTORES: int
    COMPLETITUD: str = None
    RECONSTRUCCION: str = None
    # PORCENTAJE_HUECOS: float
    # CONDUCTORES_CORREGIDOS_PARAMETROS: dict
    # ERROR_POLILINEA: float
    # ERROR_CATENARIA: float
    PUNTUACION_APRIORI: PuntuacionApriori
