from fastapi import FastAPI
from fastapi import APIRouter


app = FastAPI(title="Seer")
api_router = APIRouter()

# @app.on_event('startup')
# async def app_startup() -> None:
#     """
#     Событие вызывается когда основное приложение было запущено

#     :return: None
#     """
#     pass


# @app.on_event('shutdown')
# async def app_shutdown() -> None:
#     """
#     Событие вызывается когда основное приложение было остановлено.

#     :return: None
#     """
#     pass
