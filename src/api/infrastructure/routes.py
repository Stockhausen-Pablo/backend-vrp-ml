from flask_restful import Api
from src.api.resources import *
from src.api.models.dto.training_request_dto import MLTrainingRequestDto

ml_training_request = MLTrainingRequestDto()


def create_routes(api: Api):
    api.add_resource(StopListResource, "/stops")
    api.add_resource(StopResource, "/stops/<int:stop_id>")

    api.add_resource(MicrohubListResource, "/microhubs")
    api.add_resource(MicrohubResource, "/microhubs/<int:microhub_id>")
    
    api.add_resource(MLTrainResource, '/machine-learning/training')
