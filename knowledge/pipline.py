import json
import google.generativeai as genai
from app import app
from models.models import db, LabService


genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_metadata(test_name):

    prompt = f"""
Generate medical laboratory metadata.

Test:
{test_name}

Return JSON only:

{{
  "alias_names": [
    "alternative names",
    "abbreviations",
    "common search terms"
  ],
  "sample_type": "sample type"
}}
"""

    response = model.generate_content(prompt)

    return json.loads(response.text)



def update_lab_services():

    with app.app_context():

        services = LabService.query.filter(
            LabService.alias_names == None
        ).all()


        for service in services:

            try:
                result = generate_metadata(service.name)

                service.alias_names = ",".join(
                    result["alias_names"]
                )

                service.sample_type = result["sample_type"]

                print(
                    "Updated:",
                    service.name
                )

                db.session.commit()


            except Exception as e:

                db.session.rollback()

                print(
                    "Failed:",
                    service.name,
                    e
                )


if __name__ == "__main__":
    update_lab_services()