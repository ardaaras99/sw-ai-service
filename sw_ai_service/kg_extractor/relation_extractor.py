import asyncio
from itertools import product

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rich import print as rprint
from sw_onto_generation.base.base_node import BaseNode
from sw_onto_generation.base.base_relation import BaseRelation

from sw_ai_service.kg_extractor.utils import get_type_classes


class HasRelation(BaseModel):
    value: bool = Field(default=False, description="Bu iki node arasında bir ilişki var mı?")
    reason: str = Field(default="", description="Bu kararı vermenizin sebebi nedir?")


class RelationExtractor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def run(self, full_nodes: list[BaseNode], relation_classes_list: list[type[BaseRelation]]) -> list[BaseRelation]:
        predefined_relations = self.extract_predefined_relations(full_nodes=full_nodes, relation_classes_list=relation_classes_list)
        relations_w_llm = await self.extract_relations_w_llm(full_nodes=full_nodes, relation_classes_list=relation_classes_list)
        return predefined_relations + relations_w_llm

    def extract_predefined_relations(self, full_nodes: list[BaseNode], relation_classes_list: list[type[BaseRelation]]) -> list[BaseRelation]:
        predefined_rcl = [rel for rel in relation_classes_list if not rel.relation_config.ask_llm]

        predefined_rel_instances = []
        for rc in predefined_rcl:
            source_class = rc.model_fields["source_node"].annotation
            target_class = rc.model_fields["target_node"].annotation
            source_nodes = [node for node in full_nodes if isinstance(node, source_class)]
            target_nodes = [node for node in full_nodes if isinstance(node, target_class)]
            for source_node in source_nodes:
                for target_node in target_nodes:
                    relation = rc(
                        source_node=source_node,
                        target_node=target_node,
                        reason=f"{source_node.__class__.__name__} is related to {target_node.__class__.__name__}, created if two nodes are extracted",
                        reference_text="Predefined",
                    )
                    predefined_rel_instances.append(relation)

        return predefined_rel_instances

    async def check_relation(self, s_node: BaseNode, t_node: BaseNode, rc: type[BaseRelation]) -> HasRelation:
        system_message = """
            You are a helpful assistant that checks if two nodes have a relation.
            You are given a relation class and two nodes.
            You need to check if the two nodes have a relation with the relation class.
            Answer in Turkish.  
        """
        user_message = f"""
            Do {repr(s_node)} and {repr(t_node)} have a relation with {rc.__name__}? Answer in Turkish.
        """
        prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", user_message)])
        structured_llm = self.llm.with_structured_output(HasRelation, strict=True)
        chain = prompt | structured_llm
        return await chain.ainvoke({"s_node": s_node, "t_node": t_node, "rc": rc.__name__})

    async def extract_relations_w_llm(self, full_nodes: list[BaseNode], relation_classes_list: list[type[BaseRelation]]) -> list[BaseRelation]:
        relation_classes_to_ask_llm = [rel for rel in relation_classes_list if rel.relation_config.ask_llm]
        extracted_relations = []

        # Collect all async tasks for parallel execution
        tasks = []
        task_metadata = []

        for rc in relation_classes_to_ask_llm:
            source_types = get_type_classes(rc, "source_node")
            target_types = get_type_classes(rc, "target_node")
            possible_relations = product(source_types, target_types)
            for s_class, t_class in possible_relations:
                rprint(f"Preparing relation checks for {rc.__name__}, between {s_class.__name__} and {t_class.__name__}")
                s_nodes = [node for node in full_nodes if isinstance(node, s_class)]
                t_nodes = [node for node in full_nodes if isinstance(node, t_class)]
                for s_node in s_nodes:
                    for t_node in t_nodes:
                        if s_node == t_node:
                            continue
                        else:
                            task = self.check_relation(s_node, t_node, rc)
                            tasks.append(task)
                            task_metadata.append((s_node, t_node, rc))

        rprint(f"Executing {len(tasks)} relation checks in parallel...")

        # Execute all tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks)

            # Process results
            for (s_node, t_node, rc), has_relation in zip(task_metadata, results):  # noqa: B905
                if has_relation.value:
                    extracted_relations.append(
                        rc(
                            source_node=s_node,
                            target_node=t_node,
                            reason=has_relation.reason,
                            reference_text="Extracted from the contract with LLM",
                        )
                    )
                else:
                    rprint(has_relation.reason)

        return extracted_relations
