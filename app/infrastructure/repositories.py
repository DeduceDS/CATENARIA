# app/infrastructure/repositories.py
import json
from typing import List
from sqlalchemy import Table, Column, Integer, Float, String, JSON
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from app.application.interfaces import VanoRepository
from app.domain.models import Vano, Apoyo, Conductor, Lidar
from app.infrastructure.database import Base
from app.config import settings

vano_table = Table(
    "vanos",
    Base.metadata,
    # Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ID_VANO", String, primary_key=True),
    Column("OBJECTID_VANO_2D", Integer),
    Column("LONGITUD_2D", Float),
    Column("COORDENADA_X_INICIO", Float),
    Column("COORDENADA_Y_INICIO", Float),
    Column("COORDENADA_X_FIN", Float),
    Column("COORDEANDA_Y_FIN", Float),
    Column("APOYOS", JSON),
    Column("CONDUCTORES", JSON),
    Column("LIDAR", JSON),
    schema=settings.SCHEMA_NAME,
)


# Table class representation
# class Vano(Base):
#     __tablename__ = "vanos"
#     __table_args__ = {"schema": SCHEMA_NAME}

#     ID_VANO = Column(String, primary_key=True)
#     OBJECTID_VANO_2D = Column(Integer)
#     LONGITUD_2D = Column(Float)
#     COORDENADA_X_INICIO = Column(Float)
#     COORDENADA_Y_INICIO = Column(Float)
#     COORDENADA_X_FIN = Column(Float)
#     COORDEANDA_Y_FIN = Column(Float)
#     APOYOS = Column(JSON)
#     CONDUCTORES = Column(JSON)
#     LIDAR = Column(JSON)


class VanoRepositoryImpl(VanoRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, vano: Vano) -> None:
        vano_dict = vano.to_dict()
        query = vano_table.insert().values(
            OBJECTID_VANO_2D=vano_dict["OBJECTID_VANO_2D"],
            ID_VANO=vano_dict["ID_VANO"],
            LONGITUD_2D=vano_dict["LONGITUD_2D"],
            COORDENADA_X_INICIO=vano_dict["COORDENADA_X_INICIO"],
            COORDENADA_Y_INICIO=vano_dict["COORDENADA_Y_INICIO"],
            COORDENADA_X_FIN=vano_dict["COORDENADA_X_FIN"],
            COORDEANDA_Y_FIN=vano_dict["COORDEANDA_Y_FIN"],
            APOYOS=json.dumps(vano_dict["APOYOS"]),
            CONDUCTORES=json.dumps(vano_dict["CONDUCTORES"]),
            LIDAR=json.dumps(vano_dict["LIDAR"]),
        )
        await self.session.execute(query)
        await self.session.commit()

    async def get_all(self) -> List[Vano]:
        query = text(f"SELECT * FROM {SCHEMA_NAME}.vanos")
        result = await self.session.execute(query)
        vanos = []
        for row in result.fetchall():
            vano_dict = dict(row)
            vano_dict["APOYOS"] = [
                Apoyo(**apoyo) for apoyo in json.loads(vano_dict["APOYOS"])
            ]
            vano_dict["CONDUCTORES"] = [
                Conductor(**conductor)
                for conductor in json.loads(vano_dict["CONDUCTORES"])
            ]
            vano_dict["LIDAR"] = Lidar(**json.loads(vano_dict["LIDAR"]))
            vanos.append(Vano(**vano_dict))
        return vanos
