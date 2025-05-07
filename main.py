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
            def tokensize_and_lemmatize(text):
                lemmatizer = nltk.WordNetLemmatizer()

                words = nltk.word_tokenize(text)
