import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLu()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings = none):
            self.model = None
            self.intents_path = intents_path

            self.documents = []
            self.vocabulary = []
            self.intents = []
            self.intents_responses = []

            self.function_mappings = function_mappings

            self.x = None
            self.y = None

    @staticmethod
    def tokenize_and_lemmatizer(text):
                lemmatizer = nltk.WordNetLemmatizer()

                words = nltk.word_tokenize(text)
                words = [lemmatizer.lemmatizer(word.lower()) for word in words]

                return words

    @staticmethod
    def bag_of_words(self, words):
                return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer():

            if os.path.exists(self.intents_path):
                with open(self.intents_path, 'r') as f:
                    intents_data = json.load(f)

                intents = []
                intents_responses = []
                vocabulary = []
                documents = []

            for intent in intents_data['intents']:
                if intent['tag'] not self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

            for pattern in intent['patterns']:
                patterns_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.document.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(vocabulary))

    def prepare_data(self):
                bags = []
                indices = []

                for document in self.documents:
                    words = document[0]
                    bag = self.bag_of_words(words, self.vocabulary)

                    intent_index = self.intents.index(document[1])
                    bags.append(bag)
                    indices.append(intent_index)

                self.x = np.array(bags)
                self.y = np.array(indices)







    # chatbot = ChatbotAssistant('intents.json')
    # print(chatbot.tokenize_and_lemmatizer('run running runs ran'))
