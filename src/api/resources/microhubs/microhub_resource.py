from flask_restful import Resource

from src.datastore import MicrohubDataStore


class MicrohubResource(Resource):
    def __init__(self):
        self.microhub_db = MicrohubDataStore

    def get(self, microhub_id):
        return self.microhub_db.get_microhub_by_id(self, microhub_id)