from enum import StrEnum

from pydantic import BaseModel, Field


class LLMOptions(StrEnum):
    OPENAI_O3 = "o3-2025-04-16"
    OPENAI_O3_MINI = "o3-mini-2025-01-31"
    OPENAI_GPT4o = "gpt-4o-2024-08-06"
    OPENAI_GPT4o_MINI = "gpt-4o-mini-2024-07-18"
    OPENAI_GPT4_1_NANO = "gpt-4.1-nano"


class PDFLoaderEnum(StrEnum):
    PYPDF = "pypdf"
    UNSTRUCTURED = "unstructured"


class AgentConfig(BaseModel):
    response_model: type[BaseModel]
    model: LLMOptions
    markdown: bool
    debug_mode: bool
    description: str = Field(
        default="Sen bir classification uzmanısın, sana bir dökümanın alabileceği farklı türleri veriyorum, lütfen bunlar arasından en uygun olanı seç ve score ver, Cevaplarının türkçe olması gerekiyor."
    )
    instructions: str = Field(
        default="Sen bir classification uzmanısın, sana bir dökümanın alabileceği farklı türleri veriyorum, lütfen bunlar arasından en uygun olanı seç ve score ver, Cevaplarının türkçe olması gerekiyor.",
    )


class PDFLoaderModeEnum(StrEnum):
    SINGLE = "single"
    PAGE = "page"
