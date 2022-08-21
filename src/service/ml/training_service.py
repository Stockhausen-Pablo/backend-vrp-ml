import logging

from src.datastore import StopDataStore, MicrohubDataStore
from src.service.tf.tf_service import TensorflowTrainingService


class TrainingService:
    def __init__(self):
        self.stop_db = StopDataStore
        self.microhub_db = MicrohubDataStore

    def invoke_training_instance(self, training_request_dto):
        logging.info('invoking training instance...retrieving stops defined in problem instance')

        stops_to_retrieve = training_request_dto.problem_instance.get('stop_ids')
        stops = self.stop_db.get_stops_by_training_request(self,
                                                           training_request_dto.training_meta_data,
                                                           stops_to_retrieve)

        logging.info('Successfully retrieved stops with the length of: %s', len(stops))

        # Setting up tf instance
        ml_settings = training_request_dto.ml_settings
        microhub = self.microhub_db.get_microhub_by_id(self, training_request_dto.training_meta_data.get('microhub'))
        tensorflow_service = TensorflowTrainingService(stops, ml_settings.get('agent'), ml_settings.get('learning_rate'), microhub)

        # Setting up VRP Environment
        tensorflow_service.define_vrp_environment()
