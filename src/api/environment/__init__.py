import os

env = os.environ.get("PYTHON_ENV", "development")
port = os.environ.get("PORT", 8080)

all_environments = {
    "development": {"port": 5000, "debug": True, "swagger-url": "/api/swagger", "SQLALCHEMY_DATABASE_URI": "mysql+pymysql://root:archanw@localhost/db_vrp_ml"},
    "production": {"port": port, "debug": False, "swagger-url": None, "SQLALCHEMY_DATABASE_URI": ""}
}

environment = all_environments[env]
