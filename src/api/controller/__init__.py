from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api
from flask_marshmallow import Marshmallow

from src.api.environment import environment


class MLApiController(object):
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = environment['SQLALCHEMY_DATABASE_URI']
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        self.db = SQLAlchemy(self.app)
        self.api = Api(self.app)
        self.ma = Marshmallow(self.app)

    def run(self):
        self.app.run(
            debug=environment["debug"],
            port=environment["port"]
        )

    def add_resource(self, resource, route, **kwargs):
        self.api.add_resource(resource, route, **kwargs)


ml_api_controller = MLApiController()
