from src.datastore.schema.stop_schema import *
from src.datastore.models import Stop


class StopDataStore:
    def get_all(self):
        stops = Stop.query.all()
        return stops_schema.dump(stops)

    def get_stop_by_id(self, stop_id):
        stop = Stop.query.get_or_404(stop_id)
        return stop_schema.dump(stop)

    def get_stops_by_training_request(self, training_meta_data, stop_ids):
        if not stop_ids:
            stops = Stop.query.filter(Stop.shipper.like(training_meta_data.get('shipper')),
                                      Stop.carrier.like(training_meta_data.get('carrier')),
                                      Stop.microhub.like(training_meta_data.get('microhub'))).all()
        else:
            stops = Stop.query.filter(Stop.id.in_(stop_ids),
                                      Stop.shipper.like(training_meta_data.get('shipper')),
                                      Stop.carrier.like(training_meta_data.get('carrier')),
                                      Stop.microhub.like(training_meta_data.get('microhub'))).all()
        return stops_schema.dump(stops)
