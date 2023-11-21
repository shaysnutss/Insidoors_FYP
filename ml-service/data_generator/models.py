from sqlalchemy import Column, Integer, String, Date, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

# !! Replace placeholders before running !!
Base = declarative_base()
engine = create_engine('mysql+mysqldb://<USER>:<PASSWORD>@<HOST>:<PORT>/insidoors', echo=True)


class Employee(Base):
    __tablename__ = 'employees'
    id = Column(Integer, primary_key=True)
    firstname = Column(String(255))
    lastname = Column(String(255))
    email = Column(String(255))
    gender = Column(String(255))
    business_unit = Column(String(255))
    status = Column(String(255))
    joined_date = Column(Date)
    terminated_date = Column(Date)
    profile = Column(Integer)
    suspect = Column(Boolean)
    location = Column(String(255))

class PC_Access(Base):
    __tablename__ = 'pc_access'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    access_date_time = Column(DateTime)
    log_on_off = Column(String(255))
    machine_name = Column(String(255))
    machine_location = Column(String(255))
    suspect = Column(Integer)
    working_hours = Column(Integer, nullable=True)

class Building_Access(Base):
    __tablename__ = 'building_access'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    access_date_time = Column(DateTime)
    direction = Column(String(255))
    status = Column(String(255))
    office_location = Column(String(255))
    suspect = Column(Integer)

class Proxy_Log(Base):
    __tablename__ = 'proxy_log'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    access_date_time = Column(DateTime)
    machine_name = Column(String(255))
    url = Column(String(255))
    category = Column(String(255))
    bytes_in = Column(Integer)
    bytes_out = Column(Integer)
    suspect = Column(Integer)