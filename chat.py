import random
import json

import torch
import pandas as pd

from models.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize
from utils.dialogue_manager import DialogueManager
from sentence_transformers import SentenceTransformer
from utils.yamlparser import YamlParser


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_file = "/Projects/configs/config.yaml"
cfg = YamlParser(config_file)

model = SentenceTransformer(cfg["MODEL"]["answer_model"])


def _get_response(msg: str, msg_manager):

    _response = msg_manager.generate_answer(msg)
    
    return _response


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    msg_manager = DialogueManager(model)

    while True:
        try:
            sentence = input("You: ")
            if sentence == "quit":
                break

            resp = _get_response(sentence,msg_manager)
            print(resp)
        except Exception as e:
            print("Error!, Do not use backspace at the end of line")
            
