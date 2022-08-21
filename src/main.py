import logging

from api.controller import ml_api_controller
from src.api.infrastructure.routes import create_routes


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    create_routes(ml_api_controller)
    logging.info('Service starting...')
    ml_api_controller.run()
