from models.models import Laboratory, Page

class LabDataService:

    @staticmethod
    def get_laboratory_object(page_id):
        try:
            return Laboratory.query.join(Page).filter(Page.page_id == page_id).first()
        except Exception as e:
            print(f"[LabDataService] DB error in get_laboratory_object: {e}")
            return None

    @staticmethod
    def get_all_lab_data(page_id) -> tuple[str, str]:
        lab = LabDataService.get_laboratory_object(page_id)

        if not lab:
            return (
                "No laboratory details found.",
                "No laboratory services found.",
            )

        # Lab info
        lab_info = (
            f"Laboratory Name: {lab.name}\n"
            f"Address: {lab.address}\n"
            f"Info: {lab.info}"
        )

        # Services
        services_info = "\n\n".join(
            f"Service Name: {sv.name}\nService Description: {sv.description}\nService Price: {sv.price}"
            for sv in lab.services
        ) or "No services found."

        return lab_info, services_info

    @staticmethod
    def get_lab_info(page_id) -> str:
        lab = LabDataService.get_laboratory_object(page_id)
        if lab:
            return (
                f"Laboratory Name: {lab.name}\n"
                f"Address: {lab.address}\n"
                f"Info: {lab.info}"
            )
        return "No laboratory found for the given page ID."

    @staticmethod
    def get_services_info(page_id) -> str:
        lab = LabDataService.get_laboratory_object(page_id)
        if not lab:
            return "No services found."
        return "\n\n".join(
            f"Service Name: {sv.name}\n"
            f"Service Description: {sv.description}\n"
            f"Service Price: {sv.price}"
            for sv in lab.services
        ) or "No services found."
