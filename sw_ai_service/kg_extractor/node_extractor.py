import types

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import create_model
from rich import print as rprint
from sw_onto_generation.base.base_node import BaseNode
from sw_onto_generation.base.base_relation import BaseRelation
from sw_onto_generation.base.configs import HowToExtract
from sw_onto_generation.common.common_nodes import GeneralDocumentInfo

from sw_ai_service.kg_extractor.utils import filter_node_classes_by_case, filter_node_list, node_class_to_node_dict, node_dict_to_ontology


class NodeExtractor:
    def __init__(self, llm: ChatOpenAI, node_classes_list: list[type[BaseNode]], ontology_name: str):
        self.node_classes_list = node_classes_list
        self.llm = llm
        self.ontology_name = ontology_name

    def extract_case0_nodes(self, text: str) -> list[BaseNode]:
        case0_node_classes = filter_node_classes_by_case(HowToExtract.CASE_0, self.node_classes_list)
        case0_nodes = []
        for node_class in case0_node_classes:
            rprint("Processing node class for case 0: ", node_class)
            node_dict = node_class_to_node_dict(node_class)
            ontology = node_dict_to_ontology(node_dict)
            system_message = f"""
                Sen Türkçe metinlerinden içerisinden node(düğüm) çıkarma konusunda uzman bir yapay zeka.
                Senin görevin şu:
                - Metin içerisinden {node_class.__name__} node'unu çıkarmak.
                - Çıkan node'u doğru formatta ve doğru şekilde döndürmek.
                - Çıkan node'un doğru şekilde döndürülmesi için gerekli olan tüm bilgileri doğru şekilde döndürmek.
                Node'un descriptionı senin için çok önemli, aşağıda onu bulabilirsin. Dikkatlice oku
                Node'un descriptionı: {node_class.node_config.description}
            """
            prompt = ChatPromptTemplate.from_messages([("system", system_message), ("user", "{text}")])
            structured_llm = self.llm.with_structured_output(ontology)
            chain = prompt | structured_llm
            node_class_instances = chain.invoke({"text": text})
            rprint(node_class_instances)
            case0_nodes.append(node_class_instances)

        return filter_node_list(case0_nodes)

    def extract_case1_nodes(self, case0_nodes: list[BaseNode]):
        case1_classes = filter_node_classes_by_case(HowToExtract.CASE_1, self.node_classes_list)
        case1_nodes = []
        for case0_node in case0_nodes:
            to_be_created_class = case0_node.node_config.nodeclass_to_be_created_automatically
            if to_be_created_class in case1_classes:
                new_node = to_be_created_class(reason="Predefined", reference_text="Predefined")
                case1_nodes.append(new_node)
                case1_classes.remove(to_be_created_class)
        return case1_nodes

    def extract_case2_nodes_and_relations(self, case0_nodes: list[BaseNode]):
        case2_classes = filter_node_classes_by_case(HowToExtract.CASE_2, self.node_classes_list)
        case2_nodes = []
        case2_relations = []

        for extracted_node in case0_nodes:
            # check if any of the fields of the extracted_node is a case1_class but field can be union type also
            for field_name, field_info in type(extracted_node).model_fields.items():
                if field_info.annotation in case2_classes:
                    if extracted_node.__dict__[field_name] is None:
                        extracted_node.__dict__.pop(field_name)
                    else:
                        new_node = extracted_node.__dict__[field_name]
                        extracted_node.__dict__.pop(field_name)
                        case2_nodes.append(new_node)
                        relation_class = create_model(f"Has{field_info.annotation.__name__}", __base__=BaseRelation)
                        case2_relations.append(
                            relation_class(
                                source_node=extracted_node,
                                target_node=new_node,
                                reason=f"{new_node.__class__.__name__} is extracted from {extracted_node.__class__.__name__}",
                                reference_text="Predefined",
                            )
                        )
                elif isinstance(field_info.annotation, types.UnionType):
                    for type_in_union in field_info.annotation.__args__:
                        if type_in_union in case2_classes:
                            if extracted_node.__dict__[field_name] is None:
                                extracted_node.__dict__.pop(field_name)
                            else:
                                new_node = extracted_node.__dict__[field_name]
                                extracted_node.__dict__.pop(field_name)
                                case2_nodes.append(new_node)
                                relation_class = create_model(f"Has{type_in_union.__name__}", __base__=BaseRelation)
                                case2_relations.append(
                                    relation_class(
                                        source_node=extracted_node,
                                        target_node=new_node,
                                        reason=f"{new_node.__class__.__name__} is extracted from {extracted_node.__class__.__name__}",
                                        reference_text="Predefined",
                                    )
                                )
        return case2_nodes, case2_relations

    def extract_general_document_info(self, case0_nodes: list[BaseNode]):
        for node in case0_nodes:
            if isinstance(node, GeneralDocumentInfo):
                node.doküman_tipi = self.ontology_name

        return case0_nodes

    def run(self, text: str) -> list[BaseNode]:
        case0_nodes = self.extract_case0_nodes(text)
        case1_nodes = self.extract_case1_nodes(case0_nodes)
        case2_nodes, case2_relations = self.extract_case2_nodes_and_relations(case0_nodes)
        case0_nodes = self.extract_general_document_info(case0_nodes)

        full_nodes = case0_nodes + case1_nodes + case2_nodes

        return full_nodes, case2_relations
