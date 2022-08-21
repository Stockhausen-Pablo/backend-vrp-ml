from collections import namedtuple


class Shipper(namedtuple('Shipper', ['id', 'name'])):
    def __new__(cls, id=None, name=None):
        return super(Shipper, cls).__new__(cls, id, name)


class Carrier(namedtuple('Carrier', ['id', 'name'])):
    def __new__(cls, id=None, name=None):
        return super(Carrier, cls).__new__(cls, id, name)


class Microhub(namedtuple('Microhub', ['id', 'location'])):
    def __new__(cls, id=None, location=None):
        return super(Microhub, cls).__new__(cls, id, location)