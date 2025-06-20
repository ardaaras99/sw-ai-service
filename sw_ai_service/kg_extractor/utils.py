from datetime import datetime
from typing import Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, create_model
from sw_onto_generation.base.base_node import BaseNode
from sw_onto_generation.base.configs import HowToExtract


def node_dict_to_ontology(node_dict: dict[str, tuple[type[BaseNode], bool, str]]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for key, (node_class, cardinality, description) in node_dict.items():
        field_name = key.lower().replace("node", "") + ("_nodes" if cardinality else "_node")
        if cardinality:
            #! This is a hack to make the list type work with mypy
            fields[field_name] = (list[node_class], Field(default=None, description=description))  # type: ignore
        else:
            fields[field_name] = (node_class, Field(default=None, description=description))

    model = create_model("EntityOntology", **fields)
    return model


def filter_node_classes_by_case(case: HowToExtract, node_list: list[type[BaseNode]]) -> list[type[BaseNode]]:
    return [node for node in node_list if node.node_config.how_to_extract == case]


def node_class_to_node_dict(node_class: type[BaseNode]) -> dict[str, tuple[type[BaseNode], bool, str]]:
    return {node_class.__name__: (node_class, node_class.node_config.cardinality, node_class.node_config.description)}


def filter_node_list(node_list: list[BaseNode]) -> list[BaseNode]:
    filtered = []
    for node in node_list:
        if node is not None:
            for _, field_value in node.__dict__.items():
                if isinstance(field_value, list):
                    if len(field_value) != 0:
                        filtered.extend(field_value)

                elif field_value is not None:
                    filtered.append(field_value)
    return filtered


def get_current_date_time() -> datetime:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_type_classes(cls: type, attr: str) -> list[type[Any]]:
    hints = get_type_hints(cls)
    t = hints[attr]
    origin = get_origin(t)
    if origin is None:
        # Single type
        return [t]
    elif origin is type(None):
        # NoneType (shouldn't happen for your case)
        return []
    elif origin is Union or origin is None or origin is None:
        # Union type (Python 3.10+: | operator)
        return [arg for arg in get_args(t) if arg is not type(None)]
    else:
        # Other generics
        return list(get_args(t))
