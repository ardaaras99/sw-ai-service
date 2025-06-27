# %%
import asyncio
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


async def main():
    pdf_content_extractor_engine = PDFContentExtractorEngine(
        config=PDFContentExtractorEngineConfig(
            pdf_loader_id=PDFLoaderEnum.PYPDF,
            page_mode=PDFLoaderModeEnum.SINGLE,
            pdf_path=Path("data/yenal2.pdf"),
        ),
    )

    doc_classifier_engine = DocClassifierEngine(
        config=DocClassifierEngineConfig(
            llm_model_id=LLMOptions.OPENAI_GPT4_1_NANO,
            markdown=True,
            debug_mode=False,
        ),
    )

    # text = "This is a Legal document"
    text = pdf_content_extractor_engine.run()
    doc_clf_response = doc_classifier_engine.run(text, DIR_STRUCTURE)
    rprint(doc_clf_response)
    # %%

    lib_name = doc_clf_response.lib_name
    ontology_name = doc_clf_response.ontology_name
    node_classes_list, relation_classes_list = get_all_common_and_specific_root_classes(lib_name=lib_name, ontology_name=ontology_name)

    kg_extractor_engine = KGExtractorEngine(config=KGExtractorEngineConfig(llm_model_id=LLMOptions.OPENAI_O3_MINI))

    full_nodes, full_relations = await kg_extractor_engine.run(text, node_classes_list, relation_classes_list, ontology_name)
    kg_extractor_engine.plot_network(full_nodes, full_relations)


if __name__ == "__main__":
    asyncio.run(main())

# %%
