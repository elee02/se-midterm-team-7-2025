from PyPDF2 import PdfReader
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

def parse_pdf(file_content):
    logger.info("Parsing PDF content")
    try:
        reader = PdfReader(BytesIO(file_content))
        logger.debug(f"PDF has {len(reader.pages)} pages.")
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                logger.debug(f"Extracted text length: {len(extracted)}")
                text += extracted
        logger.info("Finished parsing PDF content")
        return text
    except Exception as e:
        logger.error(f"PDF parse error: {e}")
        return ""

def parse_text(file_content):
    logger.info("Parsing text content")
    try:
        result = file_content.decode("utf-8")
        logger.info("Finished parsing text content")
        return result
    except UnicodeDecodeError:
        try:
            result = file_content.decode("latin1")
            logger.info("Finished parsing text content")
            return result
        except Exception as e:
            logger.error(f"Text parse error: {e}")
            return ""
