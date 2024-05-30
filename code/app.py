import uvicorn
import nltk
import warnings

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from code.api import setup_routers
from code.config import settings
from code import model, similarity


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        warnings.filterwarnings("ignore")

        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

        logger.info('Service started')
        yield
    finally:
        logger.info('Service stopped')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

setup_routers(app)

if __name__ == '__main__':
    uvicorn.run('code.app:app', host='0.0.0.0', reload=True)
