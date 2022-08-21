# MYSQL

### Database

```CREATE DATABASE db_vrp_ml;```

``USE db_vrp_ml;``

### Tables

**stop**

``
CREATE TABLE stops (
id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
stopNr INT(10) NOT NULL,
location POINT NOT NULL SRID 0,
demandWeight FLOAT(10) NOT NULL,
demandVolume FLOAT(10) NOT NULL,
boxAmount INT(5) NOT NULL,
tourStopId INT(10) NOT NULL,
shipper VARCHAR(45) NOT NULL,
carrier VARCHAR(45) NOT NULL,
microhub VARCHAR(45) NOT NULL,
weekDay INT NOT NULL
)
``

**microhub**

``
CREATE TABLE microhub (
id INT(3) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
name VARCHAR(45) NOT NULL,
location POINT NOT NULL
)
``

### Insert Data

``INSERT INTO stop VALUES (null, 2, Point(13.392174,52.529564), 1, 1, 1, 1);``

``INSERT INTO microhub VALUES (null, "Adenauerplatz", Point(13.3069357,52.5013048));``