import os
from urllib.parse import quote_plus
from flask import Flask
from flask_migrate import Migrate
from flask_login import LoginManager
from dotenv import load_dotenv
from models.models import db

load_dotenv()

# ── App & Config ──────────────────────────────────────────────────────────────

app = Flask(__name__)

secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    secret_key = 'dev-secret-key-change-in-production'

db_uri = os.environ.get('SQLALCHEMY_DATABASE_URI')
if not db_uri:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
    DB_NAME = os.getenv("DB_NAME", "lab_system")
    db_uri = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

app.config['SECRET_KEY'] = secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False




db.init_app(app)
migrate = Migrate(app, db)





if __name__ == '__main__':
   with app.app_context():
    db.create_all()
        
    app.run(debug=True)