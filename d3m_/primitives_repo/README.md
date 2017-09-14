Primitive Annotation Submissions
================================

This directory contains the primitive annotation .json files required for primitive submission. Once complete, the .json files must be submitted to the following gitlab repo (following the [directory hierarchy specified](https://datadrivendiscovery.org/wiki/display/gov/Primitive+Submission+Process) by the government team):

```
https://gitlab.datadrivendiscovery.org/jpl/primitives_repo
```

Also within this directory is "prim\_val.py", which has been provided to validate each .json file using the government team's endpoint. Additionally, there is "uuid\_generator.py", which produces the appropriate uuid for a user-given name/input.

Examples
========

```python
$ python prim_val.py primitive_clustering.json
Username: user
Password: 
{
  "error": "Invalid primitive", 
  "reason": {
    "build": [
      "required field"
    ], 
    "compute_resources": [
      "required field"
    ]
  }
}
```
```python
$ python3 uuid_generator.py HelloWorld 1.0
e690898a-4bc6-3997-a636-fec5cb767d4c
```


TODO
====

Determine and add the following json fields:

1. build
2. compute\_resources
