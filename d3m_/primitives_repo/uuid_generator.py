"""
UUID generator

Example:
    $ python uuid_generator.py PrimitiveName
"""

import sys
import uuid

if __name__ == "__main__":
    NAME = sys.argv[1]
    VERSION = sys.argv[2]

    UUID = uuid.uuid3(uuid.uuid3(uuid.NAMESPACE_DNS,
                                 'datadrivendiscovery.org'), NAME+VERSION)

    print(UUID)

