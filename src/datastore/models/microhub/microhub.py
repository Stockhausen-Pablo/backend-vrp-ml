from src.api.controller import ml_api_controller
from src.datastore.models.base.geometry.point import Point

db = ml_api_controller.db


class Microhub(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    location = db.Column(Point, nullable=False)
