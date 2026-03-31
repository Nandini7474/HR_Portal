import re
import spacy

nlp = spacy.load("en_core_web_sm")


email_regex = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
phone_regex = r"\+?\d[\d\s-]{8,15}"

# Extracts name, email, and phone number from resume text

def extract_contact_details(text):

    email = re.findall(email_regex, text)
    phone = re.findall(phone_regex, text)

    name = None

    doc = nlp(text[:1000])

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    return {
        "name": name,
        "email": email[0] if email else None,
        "phone": phone[0] if phone else None
    }