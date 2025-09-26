import requests
from io import BytesIO
from pdfminer.high_level import extract_text

def extract_text_from_pdf_url(pdf_url: str) -> str:
    """
    Extracts text from a PDF located at a given URL using PDFMiner.six.

    Args:
        pdf_url (str): The URL of the PDF file.

    Returns:
        str: The extracted text from the PDF, or None if an error occurs.
    """
    try:
        # Fetch the PDF content from the URL
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Create a BytesIO object from the response content
        pdf_file = BytesIO(response.content)

        # Extract text using pdfminer.high_level.extract_text
        extracted_text = extract_text(pdf_file)
        return extracted_text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching PDF from URL: {e}")
        return None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

