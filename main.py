# %%
from pathlib import Path

from rich import print as rprint
from sw_onto_generation import DIR_STRUCTURE
from sw_onto_generation.utils import get_all_common_and_specific_root_classes

from sw_ai_service.configs import LLMOptions, PDFLoaderEnum, PDFLoaderModeEnum
from sw_ai_service.doc_classifier.engine import Engine as DocClassifierEngine
from sw_ai_service.doc_classifier.engine import EngineConfig as DocClassifierEngineConfig
from sw_ai_service.kg_extractor.engine import Engine as KGExtractorEngine
from sw_ai_service.kg_extractor.engine import EngineConfig as KGExtractorEngineConfig
from sw_ai_service.pdf_content_extractor.engine import Engine as PDFContentExtractorEngine
from sw_ai_service.pdf_content_extractor.engine import EngineConfig as PDFContentExtractorEngineConfig

pdf_content_extractor_engine = PDFContentExtractorEngine(
    config=PDFContentExtractorEngineConfig(
        pdf_loader_id=PDFLoaderEnum.PYPDF,
        page_mode=PDFLoaderModeEnum.SINGLE,
        pdf_path=Path("data/yenal2.pdf"),
    ),
)

text = pdf_content_extractor_engine.run()


doc_classifier_engine = DocClassifierEngine(
    config=DocClassifierEngineConfig(
        llm_model_id=LLMOptions.OPENAI_GPT4_1_NANO,
        markdown=True,
        debug_mode=False,
    ),
)

text = pdf_content_extractor_engine.run()

first_response, second_response = doc_classifier_engine.run(text, DIR_STRUCTURE)


rprint(f"We found that the document is about [bold green]{first_response.lib_enum_instance.name}[/bold green] and [bold green]{second_response.ontology_name.name}[/bold green] ontology")

rprint(f"Document belongs to library: [bold green]{first_response.lib_enum_instance.name}[/bold green]")
rprint(f"Reasoning: [bold green]{first_response.rationale}[/bold green]")
rprint(f"Score: [bold green]{first_response.score}[/bold green]")

lib_name = first_response.lib_enum_instance.name
ontology_name = second_response.ontology_name.name
node_classes_list, relation_classes_list = get_all_common_and_specific_root_classes(lib_name=lib_name, ontology_name=ontology_name)

kg_extractor_engine = KGExtractorEngine(
    config=KGExtractorEngineConfig(
        ontology_name=ontology_name,
        node_classes_list=node_classes_list,
        relation_classes_list=relation_classes_list,
        llm_model_id=LLMOptions.OPENAI_GPT4_1_NANO,
    ),
)

full_nodes, full_relations = kg_extractor_engine.run(text)
kg_extractor_engine.plot_network(full_nodes, full_relations)

# %%
