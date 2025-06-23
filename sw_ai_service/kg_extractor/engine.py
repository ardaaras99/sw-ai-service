from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pyvis.network import Network
from sw_onto_generation.base.base_node import BaseNode
from sw_onto_generation.base.base_relation import BaseRelation

from sw_ai_service.configs import LLMOptions
from sw_ai_service.kg_extractor.node_extractor import NodeExtractor
from sw_ai_service.kg_extractor.relation_extractor import RelationExtractor


class EngineConfig(BaseModel):
    llm_model_id: LLMOptions


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.node_extractor = NodeExtractor(llm=ChatOpenAI(model=config.llm_model_id))
        self.relation_extractor = RelationExtractor(llm=ChatOpenAI(model=config.llm_model_id))

    def run(
        self,
        text: str,
        node_classes_list: list[type[BaseNode]],
        relation_classes_list: list[type[BaseRelation]],
        ontology_name: str,
    ) -> tuple[list[BaseNode], list[BaseRelation]]:
        full_nodes, case2_relations = self.node_extractor.run(
            text=text,
            node_classes_list=node_classes_list,
            ontology_name=ontology_name,
        )
        relations = self.relation_extractor.run(
            full_nodes=full_nodes,
            relation_classes_list=relation_classes_list,
        )
        full_relations = case2_relations + relations

        return full_nodes, full_relations

    @staticmethod
    def plot_network(full_nodes: list[BaseNode], full_relations: list[BaseRelation]):
        net = Network(notebook=False, height="750px", width="1500px", directed=True)

        # Add nodes to the network
        for node in full_nodes:
            title = "\n".join([f"{k}: {v}" for k, v in node.model_dump().items()])
            net.add_node(n_id=node.node_id, label=node.__class__.__name__, title=title)

        for relation in full_relations:
            relation_name = relation.__class__.__name__
            net.add_edge(relation.source_node.node_id, relation.target_node.node_id, label=relation_name)

        net.show_buttons(filter_=["physics"])
        net.save_graph("output.html")
