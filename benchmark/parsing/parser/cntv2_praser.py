import re
import json
import yaml
import time

class Config:
    def __init__(self):
        self.model = ""
        self.use_memory_efficient_attention = False
        self.lora_adapters = []
        self.controlnets = []
        self.scheduler = ""
        self.text_encoder = ""
        self.text_encoder_2 = ""
        self.tokenizer = ""
        self.tokenizer_2 = ""
        self.transformer = ""
        self.dtype = ""
        self.quantization = ""
        self.cpu_offload_gb = 0
        self.swap_space = False
        self.spasify = False
        self.tensor_parallel_size = 0
        self.gpu_memory_utilization = 0.0
        self.pipeline_parallel_size = 0
        self.disable_custom_all_reduce = False
        self.scheduler_steps = 0
        self.scheduler_config = {}
        self.intermediate_layer_outputs = False

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)

def parse_config(filename):
    config = Config()
    current_section = None
    
    # Compile regex patterns once
    lora_adapter_pattern = re.compile(r'{name="([^"]+)", strength=([0-9.]+), path="([^"]+)"\s*}')
    controlnet_pattern = re.compile(r'{name="([^"]+)", conditioning_scale=([0-9.]+), path="([^"]+)"\s*}')
    
    with open(filename, 'r') as file:
        content = file.read()  # Read entire file at once
        lines = content.splitlines()  # Split into lines

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith("[BASIC]"):
                current_section = 'BASIC'
                continue
            elif line.startswith("[LORA_ADAPTERS]"):
                current_section = 'LORA_ADAPTERS'
                continue
            elif line.startswith("[CONTROLNETS]"):
                current_section = 'CONTROLNETS'
                continue
            elif line.startswith("[PIPELINE]"):
                current_section = 'PIPELINE'
                continue
            elif line.startswith("[QUANTIZATION]"):
                current_section = 'QUANTIZATION'
                continue
            elif line.startswith("[GPU_PARALLELISM]"):
                current_section = 'GPU_PARALLELISM'
                continue
            elif line.startswith("[SCHEDULER]"):
                current_section = 'SCHEDULER'
                continue
            elif line.startswith("[OUTPUT]"):
                current_section = 'OUTPUT'
                continue

            if current_section == 'BASIC':
                if line.startswith("model="):
                    config.model = line.split('=', 1)[1].strip()
                elif line.startswith("use_memory_efficient_attention="):
                    config.use_memory_efficient_attention = line.split('=', 1)[1].strip().lower() == 'true'

            elif current_section == 'LORA_ADAPTERS':
                match = lora_adapter_pattern.search(line)
                if match:
                    adapter = {
                        'name': match.group(1),
                        'strength': float(match.group(2)),
                        'path': match.group(3)
                    }
                    config.lora_adapters.append(adapter)

            elif current_section == 'CONTROLNETS':
                match = controlnet_pattern.search(line)
                if match:
                    controlnet = {
                        'name': match.group(1),
                        'conditioning_scale': float(match.group(2)),
                        'path': match.group(3)
                    }
                    config.controlnets.append(controlnet)

            elif current_section == 'PIPELINE':
                if line.startswith("scheduler="):
                    config.scheduler = line.split('=', 1)[1].strip()
                elif line.startswith("text_encoder="):
                    config.text_encoder = line.split('=', 1)[1].strip()
                elif line.startswith("text_encoder_2="):
                    config.text_encoder_2 = line.split('=', 1)[1].strip()
                elif line.startswith("tokenizer="):
                    config.tokenizer = line.split('=', 1)[1].strip()
                elif line.startswith("tokenizer_2="):
                    config.tokenizer_2 = line.split('=', 1)[1].strip()
                elif line.startswith("transformer="):
                    config.transformer = line.split('=', 1)[1].strip()

            elif current_section == 'QUANTIZATION':
                if line.startswith("dtype="):
                    config.dtype = line.split('=', 1)[1].strip()
                elif line.startswith("quantization="):
                    config.quantization = line.split('=', 1)[1].strip()
                elif line.startswith("cpu_offload_gb="):
                    config.cpu_offload_gb = int(line.split('=', 1)[1].strip())
                elif line.startswith("swap_space="):
                    config.swap_space = line.split('=', 1)[1].strip().lower() == 'true'
                elif line.startswith("spasify="):
                    config.spasify = line.split('=', 1)[1].strip().lower() == 'true'

            elif current_section == 'GPU_PARALLELISM':
                if line.startswith("tensor_parallel_size="):
                    config.tensor_parallel_size = int(line.split('=', 1)[1].strip())
                elif line.startswith("gpu_memory_utilization="):
                    config.gpu_memory_utilization = float(line.split('=', 1)[1].strip())
                elif line.startswith("pipeline_parallel_size="):
                    config.pipeline_parallel_size = int(line.split('=', 1)[1].strip())
                elif line.startswith("disable_custom_all_reduce="):
                    config.disable_custom_all_reduce = line.split('=', 1)[1].strip().lower() == 'true'

            elif current_section == 'SCHEDULER':
                if line.startswith("scheduler_steps="):
                    config.scheduler_steps = int(line.split('=', 1)[1].strip())
                elif line.startswith("scheduler_config="):
                    config.scheduler_config = json.loads(line.split('=', 1)[1].strip())

            elif current_section == 'OUTPUT':
                if line.startswith("intermediate_layer_outputs="):
                    config.intermediate_layer_outputs = line.split('=', 1)[1].strip().lower() == 'true'

    return config


class YamlConfig:
    def __init__(self):
        self.model = ""
        self.use_memory_efficient_attention = False
        self.lora_adapters = []
        self.controlnets = []
        self.scheduler = ""
        self.text_encoder = ""
        self.text_encoder_2 = ""
        self.tokenizer = ""
        self.tokenizer_2 = ""
        self.transformer = ""
        self.dtype = ""
        self.quantization = ""
        self.cpu_offload_gb = 0
        self.swap_space = False
        self.spasify = False
        self.tensor_parallel_size = 0
        self.gpu_memory_utilization = 0.0
        self.pipeline_parallel_size = 0
        self.disable_custom_all_reduce = False
        self.scheduler_steps = 0
        self.scheduler_config = {}
        self.intermediate_layer_outputs = False

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)

def parse_yaml_config(filename):
    with open(filename, 'r') as file:
        config_data = yaml.safe_load(file)

    config =YamlConfig()

    # Parse BASIC
    config.model = config_data['BASIC']['model']
    config.use_memory_efficient_attention = config_data['BASIC']['use_memory_efficient_attention']

    # Parse LORA_ADAPTERS
    config.lora_adapters = [
        {
            'name': adapter['name'],
            'strength': adapter['strength'],
            'path': adapter['path']
        } for adapter in config_data['LORA_ADAPTERS']
    ]

    # Parse CONTROLNETS
    config.controlnets = [
        {
            'name': controlnet['name'],
            'conditioning_scale': controlnet['conditioning_scale'],
            'path': controlnet['path']
        } for controlnet in config_data['CONTROLNETS']
    ]

    # Parse PIPELINE
    config.scheduler = config_data['PIPELINE']['scheduler']
    config.text_encoder = config_data['PIPELINE']['text_encoder']
    config.text_encoder_2 = config_data['PIPELINE']['text_encoder_2']
    config.tokenizer = config_data['PIPELINE']['tokenizer']
    config.tokenizer_2 = config_data['PIPELINE']['tokenizer_2']
    config.transformer = config_data['PIPELINE']['transformer']

    # Parse QUANTIZATION
    config.dtype = config_data['QUANTIZATION']['dtype']
    config.quantization = config_data['QUANTIZATION']['quantization']
    config.cpu_offload_gb = config_data['QUANTIZATION']['cpu_offload_gb']
    config.swap_space = config_data['QUANTIZATION']['swap_space']
    config.spasify = config_data['QUANTIZATION']['spasify']

    # Parse GPU_PARALLELISM
    config.tensor_parallel_size = config_data['GPU_PARALLELISM']['tensor_parallel_size']
    config.gpu_memory_utilization = config_data['GPU_PARALLELISM']['gpu_memory_utilization']
    config.pipeline_parallel_size = config_data['GPU_PARALLELISM']['pipeline_parallel_size']
    config.disable_custom_all_reduce = config_data['GPU_PARALLELISM']['disable_custom_all_reduce']

    # Parse SCHEDULER
    config.scheduler_steps = config_data['SCHEDULER']['scheduler_steps']
    config.scheduler_config = config_data['SCHEDULER']['scheduler_config']

    # Parse OUTPUT
    config.intermediate_layer_outputs = config_data['OUTPUT']['intermediate_layer_outputs']

    return config

def benchmark_parsers():
    toml_file = '../config_files/configv2.cnt'
    yaml_file = '../config_files/config.yaml'

    # Benchmark TOML parser
    start_time = time.time()
    toml_config = parse_config(toml_file)  # Assuming the previous TOML parser is available
    toml_duration = time.time() - start_time
    print(f"CNT Parser Time: {toml_duration:.4f} seconds")

    # Benchmark YAML parser
    start_time = time.time()
    yaml_config = parse_yaml_config(yaml_file)
    yaml_duration = time.time() - start_time
    print(f"YAML Parser Time: {yaml_duration:.4f} seconds")

if __name__ == "__main__":
    benchmark_parsers()
