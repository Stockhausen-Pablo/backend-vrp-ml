from flask_restful import Resource

from src.datastore import StopDataStore


class StopResource(Resource):
    def __init__(self):
        self.stop_db = StopDataStore

    def get(self, stop_id):
        return self.stop_db.get_stop_by_id(self, stop_id)