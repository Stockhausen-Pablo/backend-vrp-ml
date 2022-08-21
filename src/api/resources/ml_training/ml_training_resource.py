import logging

from flask_restful import Resource
from src.api.infrastructure.mapper import convert_input_to
from src.datastore import StopDataStore
from src.service import TrainingService
from src.api.models.dto.training_request_dto import MLTrainingRequestDto

training_service = TrainingService()


class MLTrainResource(Resource):
    def __init__(self):
        self.stop_db = StopDataStore

    def get(self):
        return "Not implemented"

    @convert_input_to(MLTrainingRequestDto)
    def post(self):
        logging.info('ml/training got called...Processing request')
        training_service.invoke_training_instance(self)
        print("test")
