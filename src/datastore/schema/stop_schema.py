from src.api.controller import ml_api_controller
from src.datastore.models import Stop

ma = ml_api_controller.ma


class StopSchema(ma.Schema):
    class Meta:
        fields = ("id",
                  "location",
                  "demandWeight",
                  "demandVolume",
                  "boxAmount",
                  "tourStopId",
                  "shipper",
                  "carrier",
                  "microhub",
                  "weekDay",
                  "rec_date")
        model = Stop


stop_schema = StopSchema()
stops_schema = StopSchema(many=True)
