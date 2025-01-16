import os
import json
import random
import logging
import numpy as np
from faker import Faker


class IgnoreSpecificLogs(logging.Filter):
    """Filter to ignore specific log messages from gpt api."""
    def filter(self, record):
        if "HTTP Request" in record.getMessage():
            return False
        return True


def setup_logger(verbose):
    """Configure logger"""
    logger = logging.getLogger()
    logger.handlers.clear()
    if verbose:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        # Add the filter to ignore specific logs
        handler.addFilter(IgnoreSpecificLogs())
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.CRITICAL)
        handler = logging.NullHandler()
        logger.addHandler(handler)
    


def save_json(data, file_path):
    with open(file_path, "w") as f:
        if isinstance(data, dict) or isinstance(data, list):
            json.dump(data, f, indent=4)
        elif isinstance(data, str):
            try:
                # Verify if it's a valid JSON string
                json.loads(data)
                f.write(data)
            except json.JSONDecodeError:
                raise ValueError("Provided string is not valid JSON")
        else:
            raise TypeError("Data must be a list/dictionary or a JSON string")


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def set_random_seed(seed):
    """
    seed (int): The seed value to set for random number generators.
    """
    logging.info(f"Set random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    Faker.seed(seed)
    return True


def get_api_key(args):
    """Retrieve the API key from args or environment variable."""
    api_key = args.api_key if args.api_key else os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OpenAI API key either in args or by setting the OPENAI_API_KEY environment variable.")
    return api_key


def remove_key(json_data, key_to_remove):
    if isinstance(json_data, dict):
        return {k: remove_key(v, key_to_remove) for k, v in json_data.items() if k != key_to_remove}
    elif isinstance(json_data, list):
        return [remove_key(item, key_to_remove) for item in json_data]
    else:
        return json_data


def convert_json_to_plain_text(json_data, exclude=None):
    """
    conver json data to plain text for RAG
    exclude: the keys that should not be included
    """
    
    for key_to_remove in exclude:
        json_data = remove_key(json_data, key_to_remove)
    plian_text = json.dumps(json_data, separators=(',', ':'))
    return plian_text