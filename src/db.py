#!/usr/bin/env python
# encoding: utf-8

from pinecone import Pinecone
import yaml
from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

class DB:
    def __init__(self):
        self.config = self._load_config("config.yaml")
        self._db_connect()

    def _load_config(self, config_path):
        '''
        Load configuration.
        :param config_path: Local path to configuration YAML.
        '''

        fh = open(config_path, 'r')

        return yaml.load(fh, Loader=Loader)
    
    def _db_connect(self):
        '''
        Connect to database.
        '''

        self.pc = Pinecone(self.config['api_key']['pinecone'])