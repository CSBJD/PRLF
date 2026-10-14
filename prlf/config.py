"""Load portable PRLF configuration."""
import json
from easydict import EasyDict


def get_config(dataset, path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    return EasyDict({"dataset_name": dataset, "model_name": "prlf",
                     **config["common"], **config["datasets"][dataset]})
