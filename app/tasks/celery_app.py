# app/tasks/celery_app.py
import asyncio
from celery import Celery
from app.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from app.application.services import ElectraPredictServiceImpl

celery_app = Celery(
    "electra_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
)
celery_app.conf.update(
    task_track_started=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    result_persistent=True,
    # event_queue_expires=60,  # May need adjustments depending on your task frequency
)


@celery_app.task(bind=True, name="process_electra_data", track_started=True)
def process_electra_data(self, data):
    self.update_state(state="STARTED", meta={"progress": 0})
    predict_service = ElectraPredictServiceImpl()

    # Run the coroutine in a new event loop
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(predict_service.predict_data_from_json(data))

    self.update_state(state="PROGRESS", meta={"progress": 100})
    return result
