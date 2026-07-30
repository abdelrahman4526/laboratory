from models.models import db, Laboratory
from app import app

with app.app_context():
    lab = Laboratory(
        name="My Laboratory",
        info="Main laboratory"
    )

    db.session.add(lab)
    db.session.commit()

    print(lab.id)