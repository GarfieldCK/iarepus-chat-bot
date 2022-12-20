import pandas as pd
import torch
from sentence_transformers import SentenceTransformer 
from utils.yamlparser import YamlParser
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import pipeline


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Reference : Install git lfs https://stackoverflow.com/questions/48734119/git-lfs-is-not-a-git-command-unclear
    # distil_tokenizer = AutoTokenizer.from_pretrained("kornwtp/ConGen-TinyBERT-L6")
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    tokenizer = AutoTokenizer.from_pretrained("kornwtp/ConGen-RoBERTa-base")
    

    classifier = pipeline("zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    tokenizer=tokenizer)

    # nlp = pipeline("zero-shot-classification", model=nli_model, tokenizer=distil_tokenizer)
    sequence_to_classify = "สมัครค่ายอย่างไร"
    candidate_labels = ['วิธีการสมัคร', 'เกณฑ์ในการคัดเลือก']
    ans = classifier(sequence_to_classify, candidate_labels)

    print(ans)