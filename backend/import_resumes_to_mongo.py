
import os
import sys
import asyncio

# Make sure local modules are importable (run from resume_shortlister/ dir)
sys.path.insert(0, os.path.dirname(__file__))

from bson import Binary
import motor.motor_asyncio
from parser.text_extractor import extract_text_from_pdf
from parser.contact_extractor import extract_contact_details

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = "hr_intelligence_portal"
RESUMES_DIR = os.path.join(os.path.dirname(__file__), "resumes")


 # Imports all PDF resumes from the resumes directory into MongoDB
async def import_all():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    resumes_col = client[DB_NAME]["resumes"]

    pdf_files = [f for f in os.listdir(RESUMES_DIR) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF(s) in '{RESUMES_DIR}'\n")

    inserted = 0
    skipped  = 0

    for fname in sorted(pdf_files):
        path = os.path.join(RESUMES_DIR, fname)

        # Skip if already in DB (by path)
        existing = await resumes_col.find_one({"path": path})
        if existing:
            print(f"  [SKIP]   {fname}  (already in DB)")
            skipped += 1
            continue

        # Extract text
        try:
            text = extract_text_from_pdf(path) or ""
        except Exception as e:
            print(f"  [ERROR]  {fname} — text extraction failed: {e}")
            text = ""

        # Extract contact info 
        try:
            contact = extract_contact_details(text)
        except Exception:
            contact = {}

        # Read raw bytes
        with open(path, "rb") as f:
            pdf_bytes = f.read()

        doc = {
            "filename":    fname,
            "path":        path,
            "raw_text":    text,
            "pdf_content": Binary(pdf_bytes),   # stored directly in MongoDB
            "name":        contact.get("name"),
            "email":       contact.get("email"),
            "phone":       contact.get("phone"),
           
        }

        await resumes_col.insert_one(doc)
        print(f"  [OK]     {fname}  (name={contact.get('name') or 'N/A'}, "
              f"email={contact.get('email') or 'N/A'}, "
              f"size={len(pdf_bytes)//1024}KB)")
        inserted += 1

    print(f"\nDone — {inserted} inserted, {skipped} skipped.")
    client.close()


if __name__ == "__main__":
    asyncio.run(import_all())
