# app/infrastructure/repositories.py
import json
from typing import List
from sqlalchemy import Table, Column, Integer, Float, String, JSON
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from app.application.interfaces import VanoRepository
from app.domain.models import Vano, Apoyo, Conductor, Lidar, Linea
from app.infrastructure.database import Base
from app.config import settings

vano_table = Table(
    "vanos",
    Base.metadata,
    Column("id_vano", String, primary_key=True),
    Column("objectid_vano_2d", Integer),
    Column("longitud_2d", Float),
    Column("coordenada_x_inicio", Float),
    Column("coordenada_y_inicio", Float),
    Column("coordenada_x_fin", Float),
    Column("coordenada_y_fin", Float),
    Column("apoyos", JSON),
    Column("conductores", JSON),
    Column("lidar", JSON),
    schema=settings.DB_SCHEMA,
)


class VanoRepositoryImpl(VanoRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, vano: Vano) -> None:
        vano_dict = vano.to_dict()
        query = vano_table.insert().values(
            id_vano=vano_dict["ID_VANO"],
            objectid_vano_2d=vano_dict["OBJECTID_VANO_2D"],
            longitud_2d=vano_dict["LONGITUD_2D"],
            coordenada_x_inicio=vano_dict["COORDENADA_X_INICIO"],
            coordenada_y_inicio=vano_dict["COORDENADA_Y_INICIO"],
            coordenada_x_fin=vano_dict["COORDENADA_X_FIN"],
            coordenada_y_fin=vano_dict["COORDENADA_Y_FIN"],
            apoyos=json.dumps(vano_dict["APOYOS"]),
            conductores=json.dumps(vano_dict["CONDUCTORES"]),
            lidar=json.dumps(vano_dict["LIDAR"]),
        )
        await self.session.execute(query)
        await self.session.commit()

    async def get_all(self) -> Linea:
        query = text(f"SELECT * FROM {settings.DB_SCHEMA}.vanos")
        result = await self.session.execute(query)
        vanos = []
        for row in result.fetchall():
            vano_dict = {
                "ID_VANO": row.id_vano,
                "OBJECTID_VANO_2D": row.objectid_vano_2d,
                "LONGITUD_2D": row.longitud_2d,
                "COORDENADA_X_INICIO": row.coordenada_x_inicio,
                "COORDENADA_Y_INICIO": row.coordenada_y_inicio,
                "COORDENADA_X_FIN": row.coordenada_x_fin,
                "COORDEANDA_Y_FIN": row.coordeanda_y_fin,
                "APOYOS": json.loads(row.apoyos),
                "CONDUCTORES": json.loads(row.conductores),
                "LIDAR": json.loads(row.lidar),
            }
            vano_dict["APOYOS"] = [Apoyo(**apoyo) for apoyo in vano_dict["APOYOS"]]
            vano_dict["CONDUCTORES"] = [
                Conductor(**conductor) for conductor in vano_dict["CONDUCTORES"]
            ]
            vano_dict["LIDAR"] = Lidar(**vano_dict["LIDAR"])
            vanos.append(Vano(**vano_dict))
        return Linea(vanos=vanos)

    async def get_vano(self, vano_id: str) -> Vano:
        query = text(
            f"SELECT * FROM {settings.DB_SCHEMA}.vanos WHERE id_vano = :vano_id"
        )
        result = await self.session.execute(query, {"vano_id": vano_id})
        row = result.fetchone()
        if not row:
            raise ValueError(f"Vano with id {vano_id} not found")

        vano_dict = {
            "ID_VANO": row.id_vano,
            "OBJECTID_VANO_2D": row.objectid_vano_2d,
            "LONGITUD_2D": row.longitud_2d,
            "COORDENADA_X_INICIO": row.coordenada_x_inicio,
            "COORDENADA_Y_INICIO": row.coordenada_y_inicio,
            "COORDENADA_X_FIN": row.coordenada_x_fin,
            "COORDEANDA_Y_FIN": row.coordeanda_y_fin,
            "APOYOS": json.loads(row.apoyos),
            "CONDUCTORES": json.loads(row.conductores),
            "LIDAR": json.loads(row.lidar),
        }
        vano_dict["APOYOS"] = [Apoyo(**apoyo) for apoyo in vano_dict["APOYOS"]]
        vano_dict["CONDUCTORES"] = [
            Conductor(**conductor) for conductor in vano_dict["CONDUCTORES"]
        ]
        vano_dict["LIDAR"] = Lidar(**vano_dict["LIDAR"])
        return Vano(**vano_dict)
