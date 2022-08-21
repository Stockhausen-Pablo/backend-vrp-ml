from src.api.controller import ml_api_controller
from src.datastore.models.base.geometry.point import Point

db = ml_api_controller.db


class Stop(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stopNr = db.Column(db.String(50))
    location = db.Column(Point, nullable=False)
    demandWeight = db.Column(db.Float(10), nullable=False)
    demandVolume = db.Column(db.Float(10), nullable=False)
    boxAmount = db.Column(db.Integer(), nullable=False)
    tourStopId = db.Column(db.Integer())
    shipper = db.Column(db.String(45), nullable=False)
    carrier = db.Column(db.String(45), nullable=False)
    microhub = db.Column(db.String(45), nullable=False)
    weekDay = db.Column(db.Integer(), nullable=False)
