# app/application/services.py
from app.application.interfaces import (
    ElectraDataService,
    VanoRepository,
    ElectraPredictService,
)
from app.domain.models import ElectraData
from typing import Dict
from electra_package.modules_main import fit_plot_vano_group_2


class ElectraDataServiceImpl(ElectraDataService):
    def __init__(self, vano_repository: VanoRepository):
        self.vano_repository = vano_repository

    async def process_electra_data(self, data: ElectraData) -> None:
        for vano in data.vanos:
            await self.vano_repository.save(vano)


class ElectraPredictServiceImpl(ElectraPredictService):
    def __init__(self):
        pass

    async def predict_data_from_json(self, data: Dict) -> Dict:
        import numpy as np

        # import json
        def json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (list, tuple)):
                return [json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: json_serializable(value) for key, value in obj.items()}
            else:
                return obj

        # getting required out_data destructuring the returned tuple (data, rmses, maxes, correlations)
        out_data, *others = fit_plot_vano_group_2(data)

        result = json_serializable(out_data)

        return result
