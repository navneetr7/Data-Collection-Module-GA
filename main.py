from fastapi import FastAPI, HTTPException, Depends, Security, File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
import os
import uuid
import requests
from datetime import datetime
from dotenv import load_dotenv
import logging
from supabase import create_client, Client
import json
from tenacity import retry, stop_after_attempt, wait_fixed
import traceback
import bleach
from cachetools import TTLCache
import time
from urllib.parse import unquote
import base64
from fastapi.responses import RedirectResponse

print("Starting script...")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load environment variables
    load_dotenv()
    print("Loaded .env file")

    # Environment variables
    API_KEY = os.getenv("API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET")
    ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
    ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
    ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    ZENDESK_FIELD_ID = os.getenv("ZENDESK_FIELD_ID")
    ZENDESK_CUSTOMER_ANSWERS_FIELD_ID = os.getenv("ZENDESK_CUSTOMER_ANSWERS_FIELD_ID")
    BASE_URL = os.getenv("BASE_URL", "https://example.com")
    FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "https://ga-collection-frontend.vercel.app")

    print(f"API_KEY: {API_KEY}")
    print(f"GEMINI_API_KEY: {'set' if GEMINI_API_KEY else 'missing'}")
    print(f"SUPABASE_URL: {SUPABASE_URL}")
    print(f"SUPABASE_KEY: {'set' if SUPABASE_KEY else 'missing'}")
    print(f"SUPABASE_STORAGE_BUCKET: {SUPABASE_STORAGE_BUCKET}")
    print(f"ZENDESK_SUBDOMAIN: {ZENDESK_SUBDOMAIN}")
    print(f"ZENDESK_EMAIL: {'set' if ZENDESK_EMAIL else 'missing'}")
    print(f"ZENDESK_API_TOKEN: {'set' if ZENDESK_API_TOKEN else 'missing'}")
    print(f"SLACK_WEBHOOK_URL: {'set' if SLACK_WEBHOOK_URL else 'missing'}")
    print(f"ZENDESK_FIELD_ID: {ZENDESK_FIELD_ID}")
    print(f"ZENDESK_CUSTOMER_ANSWERS_FIELD_ID: {ZENDESK_CUSTOMER_ANSWERS_FIELD_ID}")
    print(f"BASE_URL: {BASE_URL}")
    print(f"FRONTEND_BASE_URL: {FRONTEND_BASE_URL}")

    # Validate environment variables
    missing_vars = []
    if not API_KEY:
        missing_vars.append("API_KEY")
    if not GEMINI_API_KEY:
        missing_vars.append("GEMINI_API_KEY")
    if not SUPABASE_URL:
        missing_vars.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing_vars.append("SUPABASE_KEY")
    if not SUPABASE_STORAGE_BUCKET:
        missing_vars.append("SUPABASE_STORAGE_BUCKET")
    if not ZENDESK_SUBDOMAIN:
        missing_vars.append("ZENDESK_SUBDOMAIN")
    if not ZENDESK_EMAIL:
        missing_vars.append("ZENDESK_EMAIL")
    if not ZENDESK_API_TOKEN:
        missing_vars.append("ZENDESK_API_TOKEN")
    if not SLACK_WEBHOOK_URL:
        missing_vars.append("SLACK_WEBHOOK_URL")
    if not ZENDESK_FIELD_ID:
        missing_vars.append("ZENDESK_FIELD_ID")
    if not ZENDESK_CUSTOMER_ANSWERS_FIELD_ID:
        missing_vars.append("ZENDESK_CUSTOMER_ANSWERS_FIELD_ID")
    if not BASE_URL:
        missing_vars.append("BASE_URL")
    if not FRONTEND_BASE_URL:
        missing_vars.append("FRONTEND_BASE_URL")
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

    # Initialize Supabase
    print("Initializing Supabase client...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized")

    # Initialize FastAPI
    app = FastAPI()
    security = HTTPBearer()
    print("Initialized FastAPI app")

    # Pydantic models
    class WebhookPayload(BaseModel):
        ticket_id: str
        requester_email: str

    class Question(BaseModel):
        type: str
        question: str
        choices: Optional[List[str]] = None
        answer: Optional[Union[str, List[str]]] = None
        answered: bool = False

    # API key verification
    def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
        if credentials.credentials != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return credentials.credentials

    # Slack logging
    def send_slack_message(level: str, message: str, ticket_id: Optional[str] = None):
        try:
            payload = {
                "text": f"[{level.upper()}] Ticket {ticket_id or 'N/A'}: {bleach.clean(message)}"
            }
            response = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Slack message: {str(e)}")

    # Zendesk OAuth token handler
    def zendesk_basic_auth_header():
        user = f"{ZENDESK_EMAIL}/token"
        token = ZENDESK_API_TOKEN
        auth = f"{user}:{token}".encode()
        return {
            "Authorization": f"Basic {base64.b64encode(auth).decode()}",
            "Content-Type": "application/json"
        }

    # Zendesk integration
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_ticket_comments(ticket_id: str) -> List[Dict]:
        url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}/comments"
        headers = zendesk_basic_auth_header()
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to fetch ticket comments: {response.text}")
        comments = response.json().get("comments", [])
        logger.info(f"Comments for ticket {ticket_id}: {json.dumps(comments)}")
        return comments

    def update_custom_field(ticket_id: str, field_id: int, value: str):
        url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}"
        headers = zendesk_basic_auth_header()
        payload = {"ticket": {"custom_fields": [{"id": field_id, "value": value}]}}
        try:
            response = requests.put(url, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Zendesk custom field update failed: {response.status_code} {response.text}")
                send_slack_message("error", f"Failed to update custom field: {response.status_code} {response.text}", ticket_id)
                return False
            logger.info(f"Custom field updated for ticket {ticket_id}: {value}")
            return True
        except Exception as e:
            logger.error(f"Zendesk custom field update exception: {str(e)}")
            send_slack_message("error", f"Failed to update custom field: {str(e)}", ticket_id)
            return False

    def add_internal_note(ticket_id: str, note: str):
        url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}/comments"
        headers = zendesk_basic_auth_header()
        payload = {"comment": {"body": bleach.clean(note), "public": False}}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code != 201:
                logger.error(f"Zendesk note failed: {response.status_code} {response.text}")
                send_slack_message("error", f"Failed to add internal note: {response.status_code} {response.text}", ticket_id)
                return False
            logger.info(f"Internal note added for ticket {ticket_id}: {note}")
            return True
        except Exception as e:
            logger.error(f"Failed to add internal note: {str(e)}")
            send_slack_message("error", f"Failed to add internal note: {str(e)}", ticket_id)
            return False

    def update_ticket(ticket_id: str, custom_field_value: str = None, customer_answers_value: str = None, status: str = None, tags: List[str] = None, internal_note: str = None):
        if custom_field_value is not None:
            field_updated = update_custom_field(ticket_id, int(ZENDESK_FIELD_ID), custom_field_value)
            if not field_updated and not internal_note:
                internal_note = f"Magic Link Generated: {custom_field_value}\nDelete this note after sending."
        if customer_answers_value is not None:
            field_updated = update_custom_field(ticket_id, int(ZENDESK_CUSTOMER_ANSWERS_FIELD_ID), customer_answers_value)
            if not field_updated:
                logger.warning(f"Failed to update customer answers field for ticket {ticket_id}, continuing execution")
        if internal_note:
            if not add_internal_note(ticket_id, internal_note):
                logger.warning(f"Failed to add internal note for ticket {ticket_id}, continuing execution")
        if status:
            url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}"
            headers = zendesk_basic_auth_header()
            payload = {"ticket": {"status": status}}
            response = requests.put(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
        if tags is not None:
            url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}"
            headers = zendesk_basic_auth_header()
            payload = {"ticket": {"tags": tags}}
            response = requests.put(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()

    # Supabase cleanup
    def cleanup_supabase_data(ticket_id: str, client: Client):
        try:
            result = client.table("data_collection").select("uuid").eq("ticket_id", ticket_id).execute()
            uuids = [row["uuid"] for row in result.data] if result.data else []
            if uuids:
                logger.info(f"Found {len(uuids)} records to delete for ticket {ticket_id}: {uuids}")
                for uuid in uuids:
                    try:
                        files = client.storage.from_(SUPABASE_STORAGE_BUCKET).list(path=uuid)
                        for file in files:
                            file_path = f"{uuid}/{file['name']}"
                            client.storage.from_(SUPABASE_STORAGE_BUCKET).remove(file_path)
                            logger.info(f"Deleted file {file_path} for ticket {ticket_id}")
                    except Exception as e:
                        logger.error(f"Failed to delete files for UUID {uuid}: {str(e)}")
                        send_slack_message("error", f"Failed to delete files for UUID {uuid}: {str(e)}", ticket_id)
                client.table("data_collection").delete().eq("ticket_id", ticket_id).execute()
                logger.info(f"Deleted {len(uuids)} records from data_collection for ticket {ticket_id}")
                send_slack_message("info", f"Cleaned up {len(uuids)} existing form records for ticket {ticket_id}", ticket_id)
            else:
                logger.info(f"No existing records found for ticket {ticket_id}")
        except Exception as e:
            logger.error(f"Failed to clean up Supabase data for ticket {ticket_id}: {str(e)}")
            send_slack_message("error", f"Failed to clean up Supabase data: {str(e)}", ticket_id)
            raise HTTPException(status_code=500, detail=f"Failed to clean up Supabase data: {str(e)}")

    # LLM call with retry
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def call_llm(llm_type: str, api_key: str, messages: List[Dict]) -> tuple[str, int]:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        payload = {
            "contents": [{"parts": [{"text": messages[0]["content"]}]}]
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            token_count = data.get("usageMetadata", {}).get("totalTokenCount", 0)
            if not content.strip():
                raise ValueError("Empty Gemini response")
            return content, token_count
        except Exception as e:
            send_slack_message("error", f"LLM call failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    # Gemini question extraction
    async def extract_questions(comments: List[Dict], ticket_id: str) -> List[Question]:
        try:
            # Prepare full ticket context
            ticket_context = "\n".join(
                f"[Comment {c['id']} - {c['created_at']} - {'Public' if c.get('public') else 'Internal'}]: {c['body']}"
                for c in comments
            )
            # Find latest public reply
            latest_reply = None
            latest_timestamp = None
            for comment in comments:
                if comment.get("public", False):
                    timestamp = datetime.strptime(comment.get("created_at"), "%Y-%m-%dT%H:%M:%SZ")
                    if not latest_timestamp or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_reply = comment.get("body")
            # Fallback to latest comment
            if not latest_reply:
                latest_comment = max(comments, key=lambda c: c.get("created_at"), default=None)
                latest_reply = latest_comment.get("body") if latest_comment else ""

            prompt = """
            You are a support assistant. Analyze the full Zendesk ticket conversation below, focusing on the agent's last reply to the customer (highlighted). Extract specific questions or requests for the customer. For questions requiring single or multiple choice answers, identify and list the choices explicitly from the comment text (e.g., lists, comma-separated options like "Choose one: A, B, C" or "Select all: X, Y, Z"). Return as JSON with:
            - type: "text" | "single" | "multiple" | "file"
            - question: string
            - choices: [] (for single/multiple choice questions only, extracted from the comment)
            Ensure the output is valid JSON. Wrap the JSON in triple backticks (```) for clarity.
            Full Ticket Context:
            {{context}}
            Agent's Last Reply:
            {{reply}}
            """
            prompt = prompt.replace("{{context}}", bleach.clean(ticket_context)).replace("{{reply}}", bleach.clean(latest_reply or ""))
            logger.info(f"Gemini prompt for ticket {ticket_id}: {prompt}")
            messages = [{"role": "user", "content": prompt}]
            content, _ = await call_llm("gemini", GEMINI_API_KEY, messages)
            logger.info(f"Gemini raw response for ticket {ticket_id}: {content}")
            if not content.strip():
                send_slack_message("error", "Empty Gemini response", ticket_id)
                raise ValueError("Empty Gemini response")
            # Simplify JSON extraction
            content = content.strip()
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()
            elif not (content.startswith("[") or content.startswith("{")):
                send_slack_message("error", f"Invalid JSON format in Gemini response: {content}", ticket_id)
                raise ValueError(f"Invalid JSON format: {content}")
            logger.info(f"Extracted JSON content for ticket {ticket_id}: {content}")
            try:
                questions = json.loads(content)
                if not isinstance(questions, list):
                    send_slack_message("error", f"Gemini response is not a JSON array: {content}", ticket_id)
                    raise ValueError("Gemini response is not a JSON array")
                return [Question(**q) for q in questions]
            except json.JSONDecodeError as e:
                send_slack_message("error", f"Invalid JSON in Gemini response: {content}, Error: {str(e)}", ticket_id)
                raise ValueError(f"Invalid JSON: {str(e)}")
        except Exception as e:
            send_slack_message("error", f"Gemini failed: {str(e)}", ticket_id)
            raise HTTPException(status_code=500, detail=f"Failed to extract questions: {str(e)}")

    # Helper to get ticket status from Zendesk
    def get_ticket_status(ticket_id: str) -> str:
        url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets/{ticket_id}.json"
        headers = zendesk_basic_auth_header()
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.error(f"Failed to fetch ticket status: {response.status_code} {response.text}")
            raise HTTPException(status_code=500, detail="Failed to fetch ticket status")
        ticket = response.json().get("ticket", {})
        return ticket.get("status", "")

    # Endpoints
    @app.post("/data-link-start")
    async def data_link_start(payload: WebhookPayload, api_key: str = Depends(verify_api_key)):
        request_id = str(uuid.uuid4())
        try:
            ticket_id = bleach.clean(payload.ticket_id)
            requester_email = bleach.clean(payload.requester_email)
            logger.info(f"Request {request_id}: Processing ticket {ticket_id}")
            if not ticket_id.isdigit():
                send_slack_message("error", "Invalid ticket_id", ticket_id)
                raise HTTPException(status_code=400, detail="Invalid ticket_id")

            # Check ticket status before processing
            ticket_status = get_ticket_status(ticket_id)
            if ticket_status != "open":
                logger.info(f"Ticket {ticket_id} is not open (status: {ticket_status}), skipping data collection.")
                return {"status": "skipped", "reason": f"Ticket status is {ticket_status}, not open."}

            # Clean up existing data
            cleanup_supabase_data(ticket_id, supabase)

            comments = get_ticket_comments(ticket_id)
            if not comments:
                send_slack_message("error", "No comments found", ticket_id)
                raise HTTPException(status_code=400, detail="No comments found")

            questions = await extract_questions(comments, ticket_id)
            if not questions:
                send_slack_message("error", "No questions extracted", ticket_id)
                raise HTTPException(status_code=400, detail="No questions extracted")

            form_uuid = str(uuid.uuid4())
            magic_link = f"{FRONTEND_BASE_URL}/form/{form_uuid}"

            # Transaction: cleanup and insert
            try:
                cleanup_supabase_data(ticket_id, supabase)
                data = {
                    "uuid": form_uuid,
                    "ticket_id": ticket_id,
                    "customer_email": requester_email,
                    "questions": [q.model_dump() for q in questions],
                    "status": "pending",
                    "total_upload_size": 0,
                    "created_at": datetime.utcnow().isoformat() + 'Z'  # Ensure UTC with 'Z' suffix
                }
                logger.info(f"Request {request_id}: Inserting data for ticket {ticket_id}: {json.dumps(data)}")
                supabase.table("data_collection").insert(data).execute()
            except Exception as e:
                logger.error(f"Transaction failed for ticket {ticket_id}: {str(e)}")
                send_slack_message("error", f"Transaction failed: {str(e)}", ticket_id)
                raise HTTPException(status_code=500, detail=f"Transaction failed: {str(e)}")

            # Clear existing fields and tags
            update_ticket(ticket_id, custom_field_value="", customer_answers_value="", tags=[])
            update_ticket(ticket_id, custom_field_value=magic_link)
            # Set ticket status to pending after processing
            update_ticket(ticket_id, status="pending")
            send_slack_message("info", f"Magic link generated: {magic_link}", ticket_id)
            logger.info(f"Request {request_id}: Success for ticket {ticket_id}")

            return {"status": "success", "magic_link": magic_link}
        except Exception as e:
            send_slack_message("error", f"Error in data-link-start: {str(e)}", ticket_id)
            logger.error(f"Request {request_id}: Failed for ticket {ticket_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/form-data/{uuid}")
    async def get_form_data(uuid: str, api_key: str = Depends(verify_api_key)):
        try:
            uuid = bleach.clean(uuid)
            result = supabase.table("data_collection").select("*").eq("uuid", uuid).execute()
            if not result.data:
                send_slack_message("error", f"Form not found: {uuid}")
                raise HTTPException(status_code=404, detail="Form not found")
            send_slack_message("info", f"Form data accessed: {uuid}")
            return result.data[0]
        except Exception as e:
            send_slack_message("error", f"Error in get-form-data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/submit-answer/{uuid}")
    async def submit_answer(
        uuid: str,
        questions: str = Form(...),
        files: List[UploadFile] = File(None),
        api_key: str = Depends(verify_api_key)
    ):
        ticket_id = None
        try:
            uuid = bleach.clean(uuid)
            if not questions.strip():
                send_slack_message("error", "Empty questions data", ticket_id)
                raise HTTPException(status_code=400, detail="Questions data cannot be empty")
            logger.info(f"Raw questions data for UUID {uuid}: {questions}")
            try:
                submitted_questions = json.loads(questions)
                if not isinstance(submitted_questions, list):
                    send_slack_message("error", "Questions must be a JSON array", ticket_id)
                    raise HTTPException(status_code=400, detail="Questions must be a JSON array")
                submitted_questions = [Question(**q) for q in submitted_questions]
            except json.JSONDecodeError as e:
                send_slack_message("error", f"Invalid questions JSON: {str(e)}", ticket_id)
                raise HTTPException(status_code=400, detail=f"Invalid questions JSON: {str(e)}")

            result = supabase.table("data_collection").select("*").eq("uuid", uuid).execute()
            if not result.data:
                send_slack_message("error", f"Form not found: {uuid}", ticket_id)
                raise HTTPException(status_code=404, detail="Form not found")
            form_data = result.data[0]
            ticket_id = form_data["ticket_id"]

            if form_data["status"] == "completed":
                send_slack_message("error", f"Form already completed: {uuid}", ticket_id)
                raise HTTPException(status_code=400, detail="Form already completed")

            total_upload_size = form_data.get("total_upload_size", 0)
            file_urls = []
            if files:
                for file in files:
                    if file.size > 25 * 1024 * 1024:
                        send_slack_message("error", f"File too large: {file.filename}", ticket_id)
                        raise HTTPException(status_code=400, detail="File exceeds 25MB")
                    total_upload_size += file.size
                    if total_upload_size > 100 * 1024 * 1024:
                        send_slack_message("error", "Total upload size exceeds 100MB", ticket_id)
                        raise HTTPException(status_code=400, detail="Total upload size exceeds 100MB")
                    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
                        send_slack_message("error", f"Invalid file type: {file.content_type}", ticket_id)
                        raise HTTPException(status_code=400, detail="Invalid file type")

                    file_content = await file.read()
                    if not file_content or isinstance(file_content, bool):
                        send_slack_message("error", f"Invalid file content for {file.filename}", ticket_id)
                        raise HTTPException(status_code=400, detail=f"Invalid file content: {file.filename}")
                    file_path = f"{uuid}/{file.filename}"
                    # Delete existing file to avoid conflicts
                    try:
                        supabase.storage.from_(SUPABASE_STORAGE_BUCKET).remove(file_path)
                    except:
                        pass  # Ignore if file doesn't exist
                    # Upload file
                    try:
                        supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(file_path, file_content)
                    except Exception as e:
                        logger.error(f"Failed to upload file {file_path}: {str(e)}")
                        send_slack_message("error", f"Failed to upload file {file_path}: {str(e)}", ticket_id)
                        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
                    file_urls.append(file.filename)

            stored_questions = form_data["questions"]
            file_index = 0
            for i, stored_q in enumerate(stored_questions):
                submitted_q = next((q for q in submitted_questions if q.question == stored_q["question"]), None)
                if submitted_q:
                    if stored_q["type"] == "file" and file_index < len(file_urls):
                        stored_q["answer"] = file_urls[file_index]
                        file_index += 1
                    else:
                        stored_q["answer"] = submitted_q.answer
                    stored_q["answered"] = True

            all_answered = all(q["answered"] for q in stored_questions)
            status = "completed" if all_answered else "partial"

            supabase.table("data_collection").update({
                "questions": stored_questions,
                "status": status,
                "total_upload_size": total_upload_size,
                "created_at": datetime.utcnow().isoformat()  # Update created_at on submission
            }).eq("uuid", uuid).execute()

            if all_answered:
                response_link = f"{BASE_URL}/responses/{ticket_id}"
                update_ticket(ticket_id, customer_answers_value=response_link, tags=["info_collected"])
                send_slack_message("info", f"Form completed: {uuid}, Response link: {response_link}", ticket_id)

            return {"status": "success", "form_status": status}
        except HTTPException as e:
            raise e
        except Exception as e:
            send_slack_message("error", f"Error in submit-answer: {str(e)}", ticket_id)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/magic-links")
    async def get_magic_links(api_key: str = Depends(verify_api_key)):
        try:
            result = supabase.table("data_collection").select("ticket_id, customer_email, uuid, status, created_at").execute()
            if not result.data:
                send_slack_message("info", "No magic links found")
                return []
            # Construct magic_link dynamically
            links = [
                {
                    "ticket_id": item["ticket_id"],
                    "customer_email": item["customer_email"],
                    "magic_link": f"{BASE_URL}/form/{item['uuid']}",
                    "status": item["status"],
                    "created_at": item["created_at"]
                }
                for item in result.data
            ]
            send_slack_message("info", "Magic links accessed")
            return links
        except Exception as e:
            send_slack_message("error", f"Error in get-magic-links: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/ticket-responses/{ticket_id}")
    async def get_ticket_responses(ticket_id: str, api_key: str = Depends(verify_api_key)):
        try:
            ticket_id = bleach.clean(ticket_id)
            if not ticket_id.isdigit():
                send_slack_message("error", "Invalid ticket_id", ticket_id)
                raise HTTPException(status_code=400, detail="Invalid ticket_id")
            result = supabase.table("data_collection").select("*").eq("ticket_id", ticket_id).execute()
            if not result.data:
                send_slack_message("error", f"No responses found for ticket: {ticket_id}", ticket_id)
                raise HTTPException(status_code=404, detail="No responses found")
            send_slack_message("info", f"Responses accessed for ticket: {ticket_id}", ticket_id)
            return result.data
        except Exception as e:
            send_slack_message("error", f"Error in get-ticket-responses: {str(e)}", ticket_id)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/signed-download/{uuid}/{filename}")
    async def get_signed_download(uuid: str, filename: str, api_key: str = Depends(verify_api_key)):
        try:
            uuid = bleach.clean(uuid)
            filename = unquote(bleach.clean(filename))
            if not uuid or not filename:
                raise HTTPException(status_code=400, detail="UUID and filename cannot be empty")

            file_path = f"{uuid}/{filename}"
            try:
                files = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).list(path=uuid)
                logger.info(f"Files in {uuid}: {[f['name'] for f in files]}")
            except Exception as e:
                logger.error(f"Error listing files for UUID {uuid}: {str(e)}")
                send_slack_message("error", f"Error listing files for {uuid}: {str(e)}")
                raise HTTPException(status_code=500, detail="Error accessing storage")

            if not files or not any(f['name'] == filename for f in files):
                logger.error(f"File not found: {file_path}")
                send_slack_message("error", f"File not found: {file_path}")
                raise HTTPException(status_code=404, detail="File not found in storage")

            signed_url_response = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).create_signed_url(file_path, 3600)
            if not signed_url_response or 'signedURL' not in signed_url_response:
                logger.error(f"Failed to create signed URL for {file_path}. Response: {signed_url_response}")
                send_slack_message("error", f"Failed to create signed URL for {file_path}")
                raise HTTPException(status_code=404, detail="Error creating signed URL")

            signed_url = signed_url_response['signedURL']
            logger.info(f"Generated signed URL for {file_path}: {signed_url}")
            send_slack_message("info", f"Generated signed URL for {file_path}")
            try:
                response = requests.head(signed_url, timeout=5)
                if response.status_code != 200:
                    logger.error(f"Signed URL inaccessible: {signed_url}, Status: {response.status_code}")
                    send_slack_message("error", f"Signed URL inaccessible: {response.status_code}")
                    raise HTTPException(status_code=500, detail="Signed URL inaccessible")
            except Exception as e:
                logger.error(f"Error testing signed URL {signed_url}: {str(e)}")
                raise HTTPException(status_code=500, detail="Error validating signed URL")

            return {
                "signed_url": signed_url,
                "filename": filename
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            send_slack_message("error", f"Error generating signed URL for {uuid}/{filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating signed URL: {str(e)}")

    @app.post("/logout")
    async def logout():
        # For stateless API key auth, just return success. If you add sessions/cookies, clear them here.
        return {"message": "Logged out successfully"}

    @app.get("/form/{uuid}")
    def redirect_form(uuid: str):
        frontend_url = f"{FRONTEND_BASE_URL}/form/{uuid}"
        return RedirectResponse(frontend_url)

except Exception as e:
    print(f"Error starting app: {str(e)}")
    print(traceback.format_exc())
    exit(1)

if __name__ == "__main__":
    print("Running Uvicorn...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)