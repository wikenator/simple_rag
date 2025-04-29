#!/usr/bin/env python
# encoding: utf-8

import os, sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from db import DB

class RAG(DB):
    def __init__(self):
        super().__init__()
        self.index = self.get_index()
        self.vectors = []

    def get_index(self):
        '''
        Create or retrieve an index.
        :return: Retrieved index
        '''

        existing_indexes = [
            index_info["name"] for index_info in self.pc.list_indexes()
        ]

        # check if index already exists
        if self.config['index_name'] not in existing_indexes:
            # if does not exist, create index
            self.pc.create_index(
                self.config['index_name'],
                dimension=1024,
                metric='cosine',
                spec=self.config.index,
            )

            # wait for index to be initialized
            while not self.pc.describe_index(self.config['index_name']).status['ready']:
                time.sleep(1)

        index = self.pc.Index(self.config['index_name'])
        time.sleep(1)
        index.describe_index_stats()

        return index

    def create_embeddings(self, documents):
        '''
        Create embeddings based on the given documents and update the index.
        :param documents: List of dictionaries of documents with the format:
            [{"id": "id", "text": "document text"}, ...]
        :return: None
        '''

        embeddings = self.pc.inference.embed(
            model=self.config['models']['embed'],
            inputs=[d['text'] for d in documents],
            parameters={
                "input_type": "passage"
            }
        )

        for d, e in zip(documents, embeddings):
            self.vectors.append({
                "id": d['id'],
                "values": e['values'],
                "metadata": {'text': d['text']}
            })

        self.index.upsert(
            vectors=self.vectors,
            namespace="simple-rag"
        )

    def retrieve(self, query, top_k = 3):
        '''
        Simple retrieval of documents given a query.
        :param query: Query to send to the vector store.
        :param top_k: Return only the top k documents.
        :return: List of dictionaries of documents returned.
        '''

        enc_query = self.pc.inference.embed(
            model=self.config['models']['embed'],
            inputs=[query],
            parameters={
                "input_type": "query",
            }
        )

        vec = enc_query.data[0]['values']
        results = self.index.query(
            namespace='simple-rag',
            vector=vec,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
        )

        docs = [{
            "id": x['id'],
            "text": x["metadata"]['text']
        } for x in results["matches"]]

        return docs
    
    def retrieve_with_rerank(self, query, documents, top_n=3):
        '''
        Retrieval of documents with reranking, given a query.
        :param query: Query to send to the vector store.
        :param documents: List of dictionaries of documents that have already been retrieved without reranking.
        :param top_n: Return only the top n documents.
        :return: List of dictionaries of documents returned.
        '''

        results = self.pc.inference.rerank(
            model=self.config['models']['rerank'],
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=True,
        )

        docs = [{
            "id": x['document']['id'],
            "text": x["document"]['text']
        } for x in results.data]

        return docs
    
    def compare_results(self, query, top_k=3, top_n=3):
        '''
        Compare the results of simple and reranked retrieval methods.
        :param query: Query to send to the vector store.
        :param top_k: Return only the top k documents during simple retrieval.
        :param top_n: Return only the top n documents during reranked retrieval.
        :return: None
        '''

        original_docs = self.retrieve(query, top_k=top_k)
        reranked_docs = self.retrieve_with_rerank(query, original_docs, top_n)

        for orig, rerank in zip(original_docs, reranked_docs):
            if orig['id'] == rerank['id']:
                print(f"SAME:\n{rerank}\n\n---\n")

            else:
                print(f"ORIGINAL:\n{orig}\n\nRERANKED:\n{rerank}\n\n---\n")