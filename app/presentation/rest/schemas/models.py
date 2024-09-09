# app/domain/models.py
from pydantic import BaseModel, Field
from typing import List, Optional
from app.presentation.rest.schemas.examples.vano_example import VANO_JSON_EXAMPLE


class Apoyo(BaseModel):
    OBJECTID_APOYO_3D: int
    COD_APOYO: str
    COORDENADA_X: float
    COORDENADA_Y: float = Field(..., alias="COORDEANDA_Y")
    COORDENADAS_Z: List[float]

    model_config = {"populate_by_name": True}


class Conductor(BaseModel):
    """Conductor es la reconstruccion que hace el cliente de los conductores a partir de los datos LIDAR"""

    OBJECTID_VANO_3D: int
    LONGITUD_2D: float
    LONGITUD_3D: float
    VERTICES: List[List[float]]


class Lidar(BaseModel):
    CONDUCTORES: List[List[float]]
    APOYOS: List[List[float]]


class Vano(BaseModel):
    ID_LINEA: str
    OBJECTID_VANO_2D: int
    ID_VANO: str
    LONGITUD_2D: float
    COORDENADA_X_INICIO: float
    COORDENADA_Y_INICIO: float = Field(..., alias="COORDEANDA_Y_INICIO")
    COORDENADA_X_FIN: float
    COORDENADA_Y_FIN: float = Field(..., alias="COORDEANDA_Y_FIN")
    APOYOS: List[Apoyo]
    CONDUCTORES: List[Conductor]
    LIDAR: Lidar

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"examples": [VANO_JSON_EXAMPLE]},
    }

    # The to_dict method is commented out, but I'll leave it here in case you want to uncomment and use it later
    # def to_dict(self):
    #     return {
    #         "OBJECTID_VANO_2D": self.OBJECTID_VANO_2D,
    #         "ID_VANO": self.ID_VANO,
    #         "LONGITUD_2D": self.LONGITUD_2D,
    #         "COORDENADA_X_INICIO": self.COORDENADA_X_INICIO,
    #         "COORDENADA_Y_INICIO": self.COORDENADA_Y_INICIO,
    #         "COORDENADA_X_FIN": self.COORDENADA_X_FIN,
    #         "COORDENADA_Y_FIN": self.COORDENADA_Y_FIN,
    #         "APOYOS": [apoyo.model_dump() for apoyo in self.APOYOS],
    #         "CONDUCTORES": [conductor.model_dump() for conductor in self.CONDUCTORES],
    #         "LIDAR": self.LIDAR.model_dump(),
    #     }


class Linea(BaseModel):
    vanos: List[Vano]
