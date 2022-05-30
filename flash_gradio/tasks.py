from dataclasses import dataclass
from typing import List


@dataclass
class TaskMeta:
    data_module_import_path: str
    data_module_class: str
    task_import_path: str
    task_class: str
    linked_attributes: List[str]
    monitor: str
    supports_fiftyone: bool
    supports_gradio: bool


text_classification = TaskMeta(
    "flash.text",
    "TextClassificationData",
    "flash.text",
    "TextClassifier",
    ["num_classes", "labels", "multi_label"],
    "val_accuracy",
    False,
    True,
)
