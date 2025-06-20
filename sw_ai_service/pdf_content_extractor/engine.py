from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from pydantic import BaseModel

from sw_ai_service.configs import PDFLoaderEnum, PDFLoaderModeEnum


class EngineConfig(BaseModel):
    pdf_loader_id: PDFLoaderEnum
    page_mode: PDFLoaderModeEnum
    pdf_path: Path


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config

    def get_model(self, config: EngineConfig) -> PyPDFLoader | UnstructuredPDFLoader:
        if config.pdf_loader_id == PDFLoaderEnum.PYPDF:
            return PyPDFLoader(file_path=config.pdf_path, mode=config.page_mode)
        elif config.pdf_loader_id == PDFLoaderEnum.UNSTRUCTURED:
            return UnstructuredPDFLoader(file_path=config.pdf_path, mode=config.page_mode)
        else:
            raise ValueError(f"Invalid PDF loader id: {config.pdf_loader_id}")

    def run(self) -> str:
        loader = self.get_model(self.config)
        if self.config.page_mode == PDFLoaderModeEnum.SINGLE:
            content = loader.load()[0].page_content
        elif self.config.page_mode == PDFLoaderModeEnum.PAGE:
            content = " ".join([page.page_content for page in loader.load()])
        else:
            raise ValueError(f"Invalid page mode: {self.config.page_mode}")
        return content
