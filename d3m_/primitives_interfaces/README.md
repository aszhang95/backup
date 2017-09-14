Unsupervised Learning Series Modeling API
=========================================

This directory consists of UChicago's submissions to the "primitives\_interfaces" section of the D3M project. The contents of this directory are meant to be pushed to the gitlab repo that contains all primitive interfaces/APIs (note that your username must be in the following address):

```
https://github.com/username/primitives-interfaces/tree/master 
```

The repo above (as of 09/10/17) has the following hierarchy:

```
primitives-interfaces
|   .gitlab-ci.yml
|   README.md
|   setup.cfg
|
+-- primitives_interfaces
    |   unsupervised_learning_series_modeling.py
    |   ...
    +-- utils
        |   series.py 
```

The API ("unsupervised\_learning\_series\_modeling.py") belongs in the "primitives\_interfaces" directory (note that the directory has an underscore, whereas the project uses a dash). 

Also note that currently there does not exist a "utils" directory. However, in order to keep the interfaces and helper functions separate, we should create one (if it continues to not exist).
