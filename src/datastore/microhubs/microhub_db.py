from src.datastore.schema.microhub_schema import *
from src.datastore.models import Microhub


class MicrohubDataStore:
    def get_all(self):
        microhubs = Microhub.query.all()
        return microhubs_schema.dump(microhubs)

    def get_microhub_by_id(self, microhub_id):
        microhub = Microhub.query.get_or_404(microhub_id)
        return microhub_schema.dump(microhub)