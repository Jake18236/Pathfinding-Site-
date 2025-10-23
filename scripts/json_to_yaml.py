import json
import yaml

def convert_json_to_yaml(json_file, yaml_file):
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Convert to YAML and save
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def main():
    convert_json_to_yaml('../static/json/semester_info.json', '../static/yaml/semester_info.yaml')

if __name__ == "__main__":
    main()