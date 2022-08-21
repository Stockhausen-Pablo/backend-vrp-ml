from flask_restful import Resource

from src.datastore import StopDataStore


class StopListResource(Resource):
    def __init__(self):
        self.stop_db = StopDataStore

    def get(self):
        # TODO später verschiedene Conditions als arguments mitgeben
        # date_range = request.args.get('range', "")
        # other_filter = request.args.get('filter', "")
        return self.stop_db.get_all(self)
