from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import time
import numpy as np

class DistillEmbModel():

    def __init__(self, classifier: object):
        """
        Huggingface model : https://huggingface.co/facebook/bart-large-mnli
        Transformers source code : https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/bart/model.py
        Way to improve speed : https://discuss.huggingface.co/t/way-to-make-inference-zero-shot-pipeline-faster/1384

        Distill-Zero shot : https://huggingface.co/valhalla/distilbart-mnli-12-1?candidateLabels=%E0%B9%80%E0%B8%82%E0%B9%89%E0%B8%B2%E0%B8%A3%E0%B9%88%E0%B8%A7%E0%B8%A1%E0%B9%82%E0%B8%84%E0%B8%A3%E0%B8%87%E0%B8%81%E0%B8%B2%E0%B8%A3%2C+%E0%B8%A0%E0%B8%B2%E0%B8%9E%E0%B8%A3%E0%B8%A7%E0%B8%A1%E0%B9%82%E0%B8%84%E0%B8%A3%E0%B8%87%E0%B8%81%E0%B8%B2%E0%B8%A3&multiClass=true&text=%E0%B8%AA%E0%B8%A1%E0%B8%B1%E0%B8%84%E0%B8%A3%E0%B9%80%E0%B8%82%E0%B9%89%E0%B8%B2%E0%B8%A3%E0%B9%88%E0%B8%A7%E0%B8%A1%E0%B9%82%E0%B8%84%E0%B8%A3%E0%B8%87%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%AD%E0%B8%A2%E0%B9%88%E0%B8%B2%E0%B8%87%E0%B9%84%E0%B8%A3
        """

        self.classifier = classifier
        self.score_threshold = 0.50
    
    def tagging(self, sentenced :str, _labels : list)-> dict:
        """
        Return as a dictionary of {intent : prob}
        
        """
        tag_dict = {}
        ans = self.classifier(sentenced, _labels, multi_label = True)
        print(ans)
        
        for idx, s in enumerate(ans['scores']):
            if s > self.score_threshold:
                score = np.array([[s]])
                tag_dict.update({(ans['labels'])[idx] : score})

        # print(tag_dict)
        return tag_dict
        


    