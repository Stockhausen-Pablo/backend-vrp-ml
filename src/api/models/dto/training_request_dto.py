from collections import namedtuple


class VehicleDefinition(namedtuple('VehicleDefinition', ['amount_vehicles', 'capacity_weight', 'capacity_volume', 'vehicle_speed', 'stay_duration'])):
    def __new__(cls, amount_vehicles=None, capacity_weight=None, capacity_volume=None, vehicle_speed=None, stay_duration=None):
        return super(VehicleDefinition, cls).__new__(cls, amount_vehicles, capacity_weight, capacity_volume, vehicle_speed, stay_duration)


class ProblemInstance(namedtuple('ProblemInstance', ['stop_ids'])):
    def __new__(cls, stop_ids=None):
        return super(ProblemInstance, cls).__new__(cls, stop_ids)


class TrainingMetaData(namedtuple('TrainingMetaData', ['week_day', 'shipper', 'carrier', 'microhub'])):
    def __new__(cls,  week_day=None, shipper=None, carrier=None, microhub=None):
        return super(TrainingMetaData, cls).__new__(cls, week_day, shipper, carrier, microhub)


class MLSettings(namedtuple('MLSettings', ['agent', 'learning_rate'])):
    def __new__(cls, agent=None, learning_rate=None):
        return super(MLSettings, cls).__new__(cls, agent, learning_rate)


class MLTrainingRequestDto(namedtuple('MLTrainingRequestDto', ['training_meta_data', 'ml_settings', 'problem_instance', 'vehicle_definition'])):
    def __new__(cls, training_meta_data=TrainingMetaData, ml_settings=MLSettings, problem_instance=ProblemInstance, vehicle_definition=VehicleDefinition):
        return super(MLTrainingRequestDto, cls).__new__(cls, training_meta_data, ml_settings, problem_instance, vehicle_definition)
