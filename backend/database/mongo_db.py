import os
import motor.motor_asyncio
from bson import ObjectId, Binary
from datetime import datetime
from typing import List, Optional

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "hr_intelligence_portal"

class MongoDB:
    # Initialize MongoDB client and collections
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        
        # Collections
        self.jds = self.db.job_descriptions
        self.resumes = self.db.resumes
        self.shortlists = self.db.shortlists # Junction for results, scores, questions, and status

    #job_descriptions
    # Save a new job description to the database
    async def save_jd(self, jd_data: dict):
        jd_doc = {
            "title": jd_data.get("title"),
            "department": jd_data.get("department"),
            "experience": jd_data.get("experience"),
            "location": jd_data.get("location"),
            "content": jd_data.get("content"),
            "created_at": datetime.utcnow()
        }
        result = await self.jds.insert_one(jd_doc)
        return str(result.inserted_id)

    # Retrieve all job descriptions from the database
    async def get_all_jds(self):
        cursor = self.jds.find().sort("created_at", -1)
        jds = await cursor.to_list(length=1000)
        for jd in jds:
            jd["id"] = str(jd["_id"])
            del jd["_id"]
        return jds
    
    # Retrieve a job description by its ID
    async def get_jd_by_id(self, jd_id: str):
        jd = await self.jds.find_one({"_id": ObjectId(jd_id)})
        if jd:
            jd["id"] = str(jd["_id"])
            del jd["_id"]
        return jd

    #resumes
    # Get a resume by path or create a new one if it doesn't exist
    async def get_or_create_resume(self, filename: str, path: str, text: str, pdf_bytes: bytes, name: Optional[str] = None, email: Optional[str] = None, phone: Optional[str] = None):
        resume = await self.resumes.find_one({"path": path})
        if not resume:
            resume_doc = {
                "filename": filename,
                "path": path,
                "raw_text": text.strip(),
                "pdf_content": Binary(pdf_bytes),
                "name": name,
                "email": email,
                "phone": phone
                # "created_at": datetime.utcnow() # User said don't add created by/created at if they implied minimal
            }
            result = await self.resumes.insert_one(resume_doc)
            return str(result.inserted_id)
        return str(resume["_id"])

    #shortlist_results
    # Save or update a shortlist result for a candidate and job description
    async def save_shortlist_result(self, jd_id: str, resume_id: str, score: float, summary: str, jd_text: str = None):
        # Update or Insert result for this specific candidate under this JD
        query = {"jd_id": jd_id, "resume_id": resume_id}
        update = {
            "$set": {
                "score": score,
                "summary": summary,
                "updated_at": datetime.utcnow()
            },
            "$setOnInsert": {
                "status": "pending", # Initial status
                "questions": []
            }
        }
        if jd_text:
            update["$set"]["job_description"] = jd_text
        await self.shortlists.update_one(query, update, upsert=True)

    # Update the status (accepted/rejected) of a candidate for a job description
    async def update_candidate_status(self, jd_id: str, resume_id: str, status: str):
        # status should be 'accepted' or 'rejected'
        await self.shortlists.update_one(
            {"jd_id": jd_id, "resume_id": resume_id},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}}
        )

    # Save generated interview questions for a candidate and job description
    async def save_questions(self, jd_id: str, resume_id: str, questions: List[str]):
        await self.shortlists.update_one(
            {"jd_id": jd_id, "resume_id": resume_id},
            {"$set": {"questions": questions, "updated_at": datetime.utcnow()}}
        )

    # Get all shortlist results for a given job description
    async def get_results_for_jd(self, jd_id: str):
        cursor = self.shortlists.find({"jd_id": jd_id}).sort("score", -1)
        results = await cursor.to_list(length=1000)
        
        # Enrich with resume details
        for r in results:
            resume = await self.resumes.find_one({"_id": ObjectId(r["resume_id"])})
            if resume:
                r["resume_details"] = resume
                # Convert ObjectIds to strings for JSON serialisation
                r["resume_id"] = str(r["resume_id"])
                r["_id"] = str(r["_id"])
                r["resume_details"]["_id"] = str(r["resume_details"]["_id"])
        return results

    # Retrieve all candidates with status 'accepted'
    async def get_accepted_candidates(self):
        cursor = self.shortlists.find({"status": "accepted"})
        results = await cursor.to_list(length=1000)
        
        enriched = []
        for r in results:
            resume = await self.resumes.find_one({"_id": ObjectId(r["resume_id"])})
            jd = await self.jds.find_one({"_id": ObjectId(r["jd_id"])})
            if resume and jd:
                enriched.append({
                    "id": str(r["_id"]),
                    "resume_id": str(r["resume_id"]),
                    "jd_id": str(r["jd_id"]),
                    "score": r["score"],
                    "summary": r.get("summaryai", r.get("summary")),
                    "status": r["status"],
                    "questions": r.get("questions", []),
                    "name": resume.get("name") or resume["filename"],
                    "email": resume.get("email"),
                    "phone": resume.get("phone"),
                    "path": resume["path"],
                    "file_url": f"http://localhost:8000/resume/{str(resume['_id'])}",
                    "jdTitle": jd["title"],
                    "department": jd["department"],
                    "job_description": r.get("job_description", jd.get("content"))
                })
        return enriched

# Singleton instance
db = MongoDB()
