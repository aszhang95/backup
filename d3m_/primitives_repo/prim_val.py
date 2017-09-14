"""
Primitive validation script for the D3M project

Example:
    $ python3 prim_val.py path/to/json_file

Todo:
    * Comment code
"""

import sys
import json
import getpass
import requests


# REST endpoint
URL = 'https://marvin.datadrivendiscovery.org/primitives/validate'


def validate(primitive_annotation_path, username, password):
    """
    Validates a primitive_annotation.json that subscribes to the primitive
    annotation schema at https://datadrivendiscovery.org/wiki/display/gov/
    Primitive+Submission+Process via a POST request.

    Args:
        primitive_annotation_path (str): path to the .json file
        username (str): the username one uses to access the D3M project
        password (str): the password one uses to access the D3M project

    Returns:
        response (json object): response from the REST endpoint
    """
    with open(primitive_annotation_path) as json_file:
        json_ = json.load(json_file)
    headers = {'Content-type': 'application/json'}

    response = requests.post(URL, json=json_, headers=headers, auth=(username, password))
    return response

if __name__ == '__main__':
    PRIMITIVE = sys.argv[1]
    USERNAME = input("Username: ")  # requires Python3 
    PASSWORD = getpass.getpass()

    REQUEST = validate(PRIMITIVE, USERNAME, PASSWORD)
    print(REQUEST.text)

