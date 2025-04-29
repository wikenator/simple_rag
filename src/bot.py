#!/usr/bin/env python
# encoding: utf-8

import os, sys
import ollama

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from rag import RAG

class ChatBot(RAG):
    def __init__(self):
        super().__init__()

    def chat(self):
        query = input('Ask me a question: ')
        retrieved_knowledge = self.retrieve_with_rerank(query, self.retrieve(query))

        instruction_prompt = "You are a helpful chatbot. Use only the following pieces of context to answer the question. Don't make up any new information:\n"
        instruction_prompt += '\n'.join([f" - {doc['text']}" for doc in retrieved_knowledge])

        stream = ollama.chat(
            model=self.config['models']['language'],
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True,
        )

        print('Chatbot response:')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
