from flask_restful import Resource

from src.datastore import MicrohubDataStore


class MicrohubListResource(Resource):
    def __init__(self):
        self.microhub_db = MicrohubDataStore

    def get(self):
        # TODO später verschiedene Conditions als arguments mitgeben
        # date_range = request.args.get('range', "")
        # other_filter = request.args.get('filter', "")
        return self.microhub_db.get_all(self)
