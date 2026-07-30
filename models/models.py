from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, timedelta
from flask_login import UserMixin
from sqlalchemy import PrimaryKeyConstraint, ForeignKeyConstraint
from enum import Enum
db = SQLAlchemy()

class Status(Enum):
    PENDING = "Pending"
    REVIEWED = "Reviewed"
    ATTENDED = "Attended"
    NO_SHOW = "No Show"

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)

class Laboratory(db.Model):
    __tablename__ = "laboratory"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    info = db.Column(db.String(200), nullable=False)
    services = db.relationship('LabService', backref='laboratory')
    inquiries = db.relationship('Inquiry', backref='laboratory')
    branches = db.relationship('Branch', backref='laboratory', lazy=True)



class Branch(db.Model):
    __tablename__ = "branches"
    id = db.Column(db.Integer, primary_key=True)
    laboratory_id = db.Column(db.Integer, db.ForeignKey('laboratory.id'), nullable=False)   
    address = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(50))
    working_hours = db.Column(db.String(200))
   


class LabService(db.Model):
    __tablename__ = 'labservices'
    id = db.Column(db.Integer, primary_key=True)
    laboratory_id = db.Column(db.Integer, db.ForeignKey('laboratory.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    patient_instructions= db.Column(db.Text)
    durations = db.Column(db.String(100))
    keywords = db.Column(db.Text)
    alias_names= db.Column(db.Text)
    sample_type = db.Column(db.String(100))
    search_text = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )




class Platform(db.Model):
    __tablename__ = 'platforms'
    id   = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    pages = db.relationship('Page', backref='platform', lazy=True)

      
class Inquiry(db.Model):
    __tablename__ = 'inquiry'
    id = db.Column(db.Integer, primary_key=True)
    laboratory_id = db.Column(db.Integer, db.ForeignKey('laboratory.id'), nullable=False)
    prescription_img = db.Column(db.String(255), nullable=True)
    status = db.Column(db.Enum(Status), default=Status.PENDING)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    comes_from = db.Column(db.String(100))
    phone_number       = db.Column(db.String(20))
    ocr_extracted_text = db.Column(db.Text)
    confidence_score   = db.Column(db.Float)
    services_mentioned = db.Column(db.String(500))

class Page(db.Model):
    __tablename__ = 'pages'
    __table_args__ = (
        PrimaryKeyConstraint('platform_id', 'page_id'),
    )
    laboratory_id = db.Column(db.Integer, db.ForeignKey('laboratory.id'), nullable=False)
    platform_id = db.Column(db.Integer, db.ForeignKey('platforms.id'), nullable=False)
    page_id = db.Column(db.String(100), nullable=False)
    token = db.Column(db.Text, nullable=False)

    clients = db.relationship('Client', backref='page', lazy=True)

class Client(db.Model):
    __tablename__ = 'clients'
    __table_args__ = (
        PrimaryKeyConstraint('platform_id', 'page_id', 'sender_id'),
        ForeignKeyConstraint(
            ['platform_id', 'page_id'],
            ['pages.platform_id', 'pages.page_id']
        ),
    )
    platform_id = db.Column(db.Integer, nullable=False)
    page_id = db.Column(db.String(100), nullable=False)
    sender_id = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.Text)
    last_bot_message = db.Column(db.Text)
    expiration_date = db.Column(db.DateTime)

class Complaint(db.Model):
    __tablename__ = 'complaints'
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(20), nullable=False)
    complaint_text = db.Column(db.Text, nullable=False)
    status = db.Column(db.Enum(Status), default=Status.PENDING)  
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    comes_from = db.Column(db.String(100))


class Homevisit(db.Model):
    __tablename__='homevisits'
    id = db.Column(db.Integer, primary_key=True)
    reference_id = db.Column(db.String(20), unique=True, nullable=True, index=True)
    name = db.Column(db.String(120), nullable=False)
    details = db.Column(db.Text)
    date = db.Column(db.String(100))
    phone_number = db.Column(db.String(50))
    status = db.Column(db.Enum(Status), default=Status.PENDING)
    booking_time = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    comes_from = db.Column(db.String(100))
    address = db.Column(db.String(255), nullable=False)

class RequestCounter(db.Model):
    __tablename__ = 'request_counter'
    id = db.Column(db.Integer, primary_key=True)
    count = db.Column(db.Integer, default=1000)

    def decrement(self):
        db.session.query(RequestCounter).filter(
            RequestCounter.id == self.id,
            RequestCounter.count > 0
        ).update({RequestCounter.count: RequestCounter.count - 1})
        db.session.commit()    