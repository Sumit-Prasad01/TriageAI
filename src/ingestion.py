from pathlib import Path
from typing import List
from hashlib import sha256
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as DocxDocument
from src.config import get_settings
from src.vectorstore import get_vector_store


SUPPORTED = {
    ".pdf",
    ".txt",
    ".md",
    ".docx"
}