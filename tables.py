import pandas as pd
import math
from app import app
from models.models import db, LabService

df = pd.read_excel("price_list_updated.xlsx")

def clean_value(value):
    if pd.isna(value):
        return None
    return value

with app.app_context():

    for _, row in df.iterrows():

        service = LabService(
            laboratory_id=5,
            name=clean_value(row["List Code / Test Name"]),
            description="",
            price=clean_value(row["Price (EGP)"]),
            patient_instructions=clean_value(row["Patient Preparation (English)"]),
            durations=clean_value(row["Duration (Turnaround Time)"]),
            sample_type=None,
            keywords=None,
            alias_names=clean_value(row["Alternative / Alias Names"]),
            search_text=None,
            is_active=True
        )

        db.session.add(service)

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(e)

print("Done")