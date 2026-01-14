import asyncio
from main_free import SessionLocal, DocumentModel
from sqlalchemy import select


async def inject_text():
    async with SessionLocal() as db:
        # specific file or first HR file
        stmt = select(DocumentModel).where(
            DocumentModel.original_filename.contains("hr")
        )
        result = await db.execute(stmt)
        docs = result.scalars().all()

        if docs:
            doc = docs[0]
            print(f"Updating {doc.original_filename} (ID: {doc.id})")

            clean_text = """
            Roneira Inc. - Human Resources Policy Manual 2024.
            
            1. Paid Time Off (PTO) Policy:
            All full-time employees are eligible for Paid Time Off (PTO).
            - 0-2 years tenure: 15 days per year.
            - 2-5 years tenure: 20 days per year.
            - 5+ years tenure: 30 days per year.
            
            Employees must submit PTO requests at least 2 weeks in advance for leaves longer than 3 days.
            
            2. Remote Work Policy:
            Employees may work remotely up to 2 days per week with manager approval.
            """

            doc.extracted_text = clean_text
            doc.is_processed = True
            doc.processing_status = "completed"

            await db.commit()
            print("Successfully injected clean text for HR document.")
        else:
            print("No HR documents found to update.")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(inject_text())
