"""
TOI Spanish Report Translation Service
Hosted on Zoho Catalyst AppSail

This Flask app receives webhooks from Zoho CRM, fetches a Scribeware
inspection report, translates the text to Spanish using Claude API,
uploads the translated file to Zoho WorkDrive, and updates the CRM record.
"""

import os
import json
import re
import time
import logging
from datetime import datetime

import requests
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup, NavigableString, Comment
import anthropic

# ---------------------------------------------------------------------------
# Configuration — these come from Catalyst environment variables
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID", "")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET", "")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN", "")
WORKDRIVE_FOLDER_ID = os.getenv("WORKDRIVE_FOLDER_ID", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "toi-translate-2026")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("toi-translate")

# Cache for Zoho access token
_access_token_cache = {"token": None, "expires_at": 0}


# ===========================================================================
# ZOHO AUTH
# ===========================================================================
def get_zoho_access_token():
    """Get a fresh Zoho access token using the refresh token."""
    now = time.time()
    if _access_token_cache["token"] and now < _access_token_cache["expires_at"] - 60:
        return _access_token_cache["token"]

    logger.info("Refreshing Zoho access token...")
    resp = requests.post(
        "https://accounts.zoho.com/oauth/v2/token",
        data={
            "refresh_token": ZOHO_REFRESH_TOKEN,
            "client_id": ZOHO_CLIENT_ID,
            "client_secret": ZOHO_CLIENT_SECRET,
            "grant_type": "refresh_token",
        },
    )
    data = resp.json()
    if "access_token" not in data:
        logger.error(f"Failed to refresh token: {data}")
        raise Exception(f"Token refresh failed: {data}")

    _access_token_cache["token"] = data["access_token"]
    _access_token_cache["expires_at"] = now + data.get("expires_in", 3600)
    logger.info("Zoho access token refreshed successfully")
    return _access_token_cache["token"]


# ===========================================================================
# SCRIBEWARE HTML FETCHING
# ===========================================================================
def fetch_scribeware_html(url):
    """Fetch the published Scribeware report HTML."""
    logger.info(f"Fetching Scribeware report from: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    logger.info(f"Fetched {len(resp.text)} characters of HTML")
    return resp.text


# ===========================================================================
# HTML TEXT EXTRACTION & REPLACEMENT
# ===========================================================================
def extract_translatable_text(html_content):
    """
    Parse HTML and extract text nodes that should be translated.
    Returns the soup object and a list of (element, original_text) tuples.
    
    We skip:
    - <script> and <style> tags
    - HTML comments
    - Strings that are only whitespace, numbers, or punctuation
    - Very short strings (1-2 chars) that are likely formatting artifacts
    """
    soup = BeautifulSoup(html_content, "html.parser")
    text_nodes = []
    
    # Tags whose content should NOT be translated
    skip_tags = {"script", "style", "code", "pre", "svg", "math"}
    
    # Pattern for strings that don't need translation
    skip_pattern = re.compile(r'^[\s\d\W]*$')
    
    for element in soup.descendants:
        if isinstance(element, NavigableString) and not isinstance(element, Comment):
            # Skip if inside a non-translatable tag
            parent = element.parent
            if parent and parent.name in skip_tags:
                continue
            
            text = str(element)
            stripped = text.strip()
            
            # Skip empty, whitespace-only, or non-word strings
            if not stripped or len(stripped) <= 2 or skip_pattern.match(stripped):
                continue
            
            # Skip strings that are just numbers (addresses, measurements, etc.)
            if re.match(r'^[\d,.\-\s/]+$', stripped):
                continue
            
            text_nodes.append((element, text))
    
    logger.info(f"Found {len(text_nodes)} translatable text nodes")
    return soup, text_nodes


def build_translation_payload(text_nodes):
    """
    Build a structured payload for Claude to translate.
    Each text node gets a numbered marker so we can map translations back.
    """
    lines = []
    for i, (element, text) in enumerate(text_nodes):
        # Preserve leading/trailing whitespace info
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def apply_translations(text_nodes, translated_map):
    """Replace original text nodes with translated text."""
    replaced_count = 0
    for i, (element, original_text) in enumerate(text_nodes):
        key = str(i)
        if key in translated_map:
            translated = translated_map[key]
            # Preserve original leading/trailing whitespace
            leading = original_text[:len(original_text) - len(original_text.lstrip())]
            trailing = original_text[len(original_text.rstrip()):]
            element.replace_with(NavigableString(leading + translated.strip() + trailing))
            replaced_count += 1
    
    logger.info(f"Replaced {replaced_count} of {len(text_nodes)} text nodes")
    return replaced_count


# ===========================================================================
# CLAUDE TRANSLATION
# ===========================================================================
TRANSLATION_SYSTEM_PROMPT = """You are a professional translator specializing in home inspection reports. 
You translate English to Spanish (Latin American standard).

CRITICAL RULES:
1. Translate ONLY the text content. Never modify HTML tags, attributes, URLs, or image references.
2. Use professional, formal Spanish appropriate for official property inspection documents.
3. Use correct technical terminology for construction, plumbing, electrical, HVAC, and roofing terms.
4. Preserve the original tone — if the English is a warning or safety concern, the Spanish should convey the same urgency.
5. Keep proper nouns (names of people, company names, brand names) unchanged.
6. Keep measurements in their original format (do not convert units).
7. Keep addresses exactly as they are in English.
8. If a phrase is a standard home inspection modifier (like "Inspected", "Not Inspected", "Not Present"), use the standard Spanish equivalents:
   - Inspected = Inspeccionado
   - Not Inspected = No Inspeccionado  
   - Not Present = No Presente
   - Functional = Funcional
   - Defective = Defectuoso
   - Safety Concern = Preocupación de Seguridad
   - Maintenance Item = Elemento de Mantenimiento
   - Repair or Replace = Reparar o Reemplazar
   - Monitor = Monitorear
   - Further Evaluation = Evaluación Adicional

RESPONSE FORMAT:
You will receive numbered text lines like [0] some text, [1] more text.
Respond with a JSON object where keys are the line numbers and values are the translations.
Example: {"0": "texto traducido", "1": "más texto"}

Respond ONLY with the JSON object — no markdown, no backticks, no explanation."""


def translate_with_claude(text_payload):
    """Send text to Claude API for translation and get back a mapping."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Split into chunks if very large (Claude handles 200K tokens, but smaller
    # chunks are faster and more reliable)
    lines = text_payload.split("\n")
    chunk_size = 500  # lines per chunk
    all_translations = {}
    
    for chunk_start in range(0, len(lines), chunk_size):
        chunk_lines = lines[chunk_start:chunk_start + chunk_size]
        chunk_text = "\n".join(chunk_lines)
        
        logger.info(f"Translating chunk: lines {chunk_start} to {chunk_start + len(chunk_lines)}")
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            system=TRANSLATION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the following text nodes to Spanish. Respond with ONLY a JSON object.\n\n{chunk_text}"
                }
            ],
        )
        
        response_text = message.content[0].text.strip()
        
        # Clean up response — remove markdown backticks if present
        if response_text.startswith("```"):
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        try:
            chunk_translations = json.loads(response_text)
            all_translations.update(chunk_translations)
            logger.info(f"Got {len(chunk_translations)} translations from this chunk")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.error(f"Response was: {response_text[:500]}")
            # Continue with what we have rather than failing entirely
    
    logger.info(f"Total translations: {len(all_translations)}")
    return all_translations


# ===========================================================================
# WORKDRIVE UPLOAD
# ===========================================================================
def upload_to_workdrive(html_content, filename):
    """Upload the translated HTML file to Zoho WorkDrive and return a share link."""
    token = get_zoho_access_token()
    
    # Step 1: Upload the file
    logger.info(f"Uploading {filename} to WorkDrive folder {WORKDRIVE_FOLDER_ID}")
    
    upload_resp = requests.post(
        "https://workdrive.zoho.com/api/v1/upload",
        headers={
            "Authorization": f"Bearer {token}",
        },
        data={
            "parent_id": WORKDRIVE_FOLDER_ID,
            "override-name-exist": "true",
        },
        files={
            "content": (filename, html_content.encode("utf-8"), "text/html"),
        },
    )
    
    if upload_resp.status_code not in (200, 201):
        logger.error(f"WorkDrive upload failed: {upload_resp.status_code} {upload_resp.text}")
        raise Exception(f"WorkDrive upload failed: {upload_resp.text}")
    
    upload_data = upload_resp.json()
    # The response structure for upload is a list of uploaded files
    if isinstance(upload_data, dict) and "data" in upload_data:
        file_info = upload_data["data"]
        if isinstance(file_info, list):
            file_id = file_info[0]["attributes"]["resource_id"]
        else:
            file_id = file_info.get("id") or file_info["attributes"]["resource_id"]
    else:
        logger.error(f"Unexpected upload response: {upload_data}")
        raise Exception(f"Unexpected upload response format")
    
    logger.info(f"File uploaded with ID: {file_id}")
    
    # Step 2: Create an external share link
    logger.info("Creating external share link...")
    
    link_payload = {
        "data": {
            "attributes": {
                "resource_id": file_id,
                "link_name": filename,
                "request_user_data": False,
                "allow_download": True,
                "role_id": "7",  # 7 = view only
            },
            "type": "links",
        }
    }
    
    link_resp = requests.post(
        "https://workdrive.zoho.com/api/v1/links",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/json",
        },
        json=link_payload,
    )
    
    if link_resp.status_code not in (200, 201):
        logger.error(f"Share link creation failed: {link_resp.status_code} {link_resp.text}")
        raise Exception(f"Share link creation failed: {link_resp.text}")
    
    link_data = link_resp.json()
    share_link = link_data["data"]["attributes"]["link"]
    logger.info(f"Share link created: {share_link}")
    
    return share_link


# ===========================================================================
# CRM UPDATE
# ===========================================================================
def update_crm_inspection(inspection_id, spanish_report_url):
    """Update the CRM Inspection record with the Spanish report link."""
    token = get_zoho_access_token()
    
    logger.info(f"Updating CRM inspection {inspection_id} with Spanish report URL")
    
    update_data = {
        "data": [
            {
                "id": inspection_id,
                "Spanish_Report": spanish_report_url,
                "Spanish_Report_Complete": True,
            }
        ]
    }
    
    resp = requests.put(
        f"https://www.zohoapis.com/crm/v2/Inspections/{inspection_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=update_data,
    )
    
    if resp.status_code != 200:
        logger.error(f"CRM update failed: {resp.status_code} {resp.text}")
        raise Exception(f"CRM update failed: {resp.text}")
    
    result = resp.json()
    logger.info(f"CRM updated successfully: {result}")
    return result


# ===========================================================================
# MAIN TRANSLATION PIPELINE
# ===========================================================================
def translate_report(inspection_id, scribeware_url, address, inspection_date):
    """
    Full pipeline:
    1. Fetch Scribeware HTML
    2. Extract translatable text
    3. Translate via Claude
    4. Rebuild HTML with Spanish text
    5. Upload to WorkDrive
    6. Update CRM
    """
    logger.info(f"=== Starting translation for inspection {inspection_id} ===")
    logger.info(f"Address: {address}")
    logger.info(f"Scribeware URL: {scribeware_url}")
    
    # Step 1: Fetch the report
    html_content = fetch_scribeware_html(scribeware_url)
    
    # Step 2: Extract text
    soup, text_nodes = extract_translatable_text(html_content)
    
    if not text_nodes:
        logger.warning("No translatable text found in report!")
        return {"status": "error", "message": "No translatable text found"}
    
    # Step 3: Build translation payload and translate
    payload = build_translation_payload(text_nodes)
    translations = translate_with_claude(payload)
    
    # Step 4: Apply translations back to HTML
    apply_translations(text_nodes, translations)
    translated_html = str(soup)
    
    # Step 5: Generate filename
    # Clean address for filename
    safe_address = re.sub(r'[^\w\s-]', '', address or "report").strip()
    safe_address = re.sub(r'\s+', '_', safe_address)
    date_str = inspection_date or datetime.now().strftime("%Y-%m-%d")
    filename = f"{safe_address}_{date_str}_Spanish.html"
    
    # Step 6: Upload to WorkDrive
    share_link = upload_to_workdrive(translated_html, filename)
    
    # Step 7: Update CRM
    update_crm_inspection(inspection_id, share_link)
    
    logger.info(f"=== Translation complete for {inspection_id} ===")
    return {
        "status": "success",
        "inspection_id": inspection_id,
        "share_link": share_link,
        "filename": filename,
        "text_nodes_translated": len(translations),
    }


# ===========================================================================
# FLASK ROUTES
# ===========================================================================
@app.route("/", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "running",
        "service": "TOI Spanish Translation",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/translate", methods=["POST"])
def handle_translate():
    """
    Webhook endpoint that CRM calls to trigger a translation.
    Expects JSON body with:
    - inspection_id
    - scribeware_url
    - address (optional)
    - inspection_date (optional)
    """
    try:
        data = request.get_json(force=True)
        logger.info(f"Received translation request: {json.dumps(data, indent=2)}")
        
        inspection_id = data.get("inspection_id")
        scribeware_url = data.get("scribeware_url")
        address = data.get("address", "")
        inspection_date = data.get("inspection_date", "")
        
        if not inspection_id or not scribeware_url:
            return jsonify({
                "status": "error",
                "message": "Missing required fields: inspection_id, scribeware_url"
            }), 400
        
        # Run the translation pipeline
        result = translate_report(inspection_id, scribeware_url, address, inspection_date)
        return jsonify(result), 200
        
    except Exception as e:
        logger.exception(f"Translation failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


# ===========================================================================
# APP ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    port = int(os.getenv("X_ZOHO_CATALYST_LISTEN_PORT", 9000))
    app.run(host="0.0.0.0", port=port, debug=False)
