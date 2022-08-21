from src.api.controller import ml_api_controller
from src.datastore.models import Microhub

ma = ml_api_controller.ma


class MicrohubSchema(ma.Schema):
    class Meta:
        fields = ("id",
                  "name",
                  "location")
        model = Microhub


microhub_schema = MicrohubSchema()
microhubs_schema = MicrohubSchema(many=True)
