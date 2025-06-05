from typing import Any
import yaml

import marko

# Calculate age
def calculate_age(birthdate, timestamp):
    age = timestamp.year - birthdate.year
    # Adjust for cases where the birthdate hasn't occurred yet this year
    if (timestamp.month, timestamp.day) < (birthdate.month, birthdate.day):
        age -= 1
    return age

def parse_json(text: str) -> dict[str, Any] | None:
    result_json = parse_json_markdown(text)

    if not result_json:
        result_json = parse_json_str(text)
    return result_json

def parse_json_markdown(text: str) -> dict[str, Any] | None:
    ast = marko.parse(text)

    for child in ast.children:
        if hasattr(child, "lang") and child.lang.lower() == "json":
            json_str = child.children[0].children
            return parse_json_str(json_str)

    return None

def parse_json_str(text: str) -> dict[str, Any] | None:
    try:
        result_json = yaml.safe_load(text)
    except yaml.parser.ParserError:
        return None

    return result_json