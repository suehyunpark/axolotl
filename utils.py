from datasets import load_dataset
import json
import csv
import os
import logging
from tqdm import tqdm

log_root = "/mnt/nas/suehyun/mpa/logs"


def get_load_func(file):
    
    def load_hf_dataset(dataset_name, split="train"):
        dataset = load_dataset(dataset_name, split=split)
        for d in tqdm(dataset):
            yield d

    def load_json(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        for d in tqdm(data):
            yield d

    def load_jsonl(file_path):
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
                                    
    def load_csv(file_path):
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for line in reader:
                yield line
                
    if file.endswith(".json"):
        load_func = load_json
    elif file.endswith(".jsonl"):
        load_func = load_jsonl
    elif file.endswith(".csv"):
        load_func = load_csv
    else:
        load_func = load_hf_dataset
        
    return load_func
        
        
def get_save_func(file):
               
    def save_json(data, file):
        with open(file, "w") as f:
            json.dump(data, f)

    def save_jsonl(data, file):
        with open(file, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n") 
                
    if file.endswith(".json"):
        save_func = save_json
    elif file.endswith(".jsonl"):
        save_func = save_jsonl
    else:
        raise ValueError(f"Invalid output file format {file}")   
    
    return save_func

            
def set_logger(input_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    if input_file:
        # Create a file handler
        log_dir = os.path.join(log_root, os.path.splitext(os.path.basename(__file__))[0])
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, os.path.basename(input_file).replace(".json", ".log"))
        print(f"Logging to {log_file}")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        
    logger.addHandler(console_handler)
    
    return logger

def pricing_info(model):
    if model.startswith("gpt-4") and model.endswith("preview"):
        input_rate = 0.01
        output_rate = 0.03
    elif model == "gpt-4":
        input_rate = 0.03
        output_rate = 0.06
    elif model == "gpt-4-32k":
        input_rate = 0.06
        output_rate = 0.12
    elif model == "gpt-3.5-turbo-0125":
        input_rate = 0.0005
        output_rate = 0.0015
    elif model == "gpt-3.5-turbo-instruct":
        input_rate = 0.0015
        output_rate = 0.0020
    else:
        raise ValueError(f"Model {model} not supported.")
    return input_rate, output_rate
