from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


class DistillEmbModel():

    def __init__(self, nli_model : object, tokenizer: object):

        self.model = nli_model
        self.tokenizer = tokenizer
        self.classifier = pipeline("zero-shot-classification",
                    model=nli_model,
                    tokenizer=tokenizer)
        self.tag_dict =  {}
        self.score_threshold = 0.45
    
    def tagging(self, sentenced :str, _labels : list)-> dict:
        """
        Return as a dictionary of {intent : prob}
        
        """
        ans = self.classifier(sentenced, _labels, multi_class = True)
        
        print(ans)
        for idx, s in enumerate(ans['scores']):
            if s > self.score_threshold:
                self.tag_dict.update({(ans['labels'])[idx] : s})

        print(self.tag_dict)
        return self.tag_dict
        


    