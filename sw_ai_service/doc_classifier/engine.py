import enum

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field

from sw_ai_service.configs import LLMOptions

INSTRUCTIONS = """
    Sen bir classification uzmanısın, sana bir dökümanın alabileceği 
    farklı türleri veriyorum, lütfen bunlar arasından en uygun olanı seç ve score ver. Vereceğin puan 0-100 arasında olmalı.
    100 maksimum skor, 0 minimum skor.
    Cevaplarının türkçe olması gerekiyor. Verdiğin score için sağlam bir gerekçe vermen gerekiyor. Detaylıca yazmaya çalış.
    """


class EngineConfig(BaseModel):
    llm_model_id: LLMOptions
    markdown: bool
    debug_mode: bool
    description: str = Field(default="We are running structured output task for classification")
    instructions: str = Field(default=INSTRUCTIONS)


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.agent = Agent(
            model=OpenAIChat(id=config.llm_model_id),
            description=config.description,
            instructions=config.instructions,
            markdown=config.markdown,
            debug_mode=config.debug_mode,
        )

    def run(self, text: str, dir_structure: dict):
        # first step
        possible_lib_names = [key for key in dir_structure.keys()]
        lib_enum: type[enum.StrEnum] = enum.StrEnum("LibEnum", possible_lib_names)

        class LibClassificationResponse(BaseModel):
            lib_enum_instance: lib_enum
            score: int = Field(description="describes how confident the model is about the document type", ge=0, le=100)
            rationale: str = Field(description="sence bu döküman neden senin seçtiğin türe ait")

        self.agent.response_model = LibClassificationResponse

        first_response: LibClassificationResponse = self.agent.run(message=text).content
        lib_name = first_response.lib_enum_instance.name

        # second step
        possible_ontology_names = [key for key in dir_structure[lib_name]]
        ontology_name_enum: type[enum.StrEnum] = enum.StrEnum("OntologyNameEnum", possible_ontology_names)

        class OntologyClassificationResponse(BaseModel):
            ontology_name: ontology_name_enum
            score: int = Field(description="describes how confident the model is about the ontology name", ge=0, le=100)
            rationale: str = Field(description="sence bu döküman neden senin seçtiğin türe ait")

        self.agent.response_model = OntologyClassificationResponse
        second_response: OntologyClassificationResponse = self.agent.run(message=text).content
        return first_response, second_response
