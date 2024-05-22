import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

class InferlessPythonModel:
    def initialize(self):
        # Load the model and tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v1-tiny-en")
        self.model = AutoModelForSequenceClassification.from_pretrained("jinaai/jina-reranker-v1-tiny-en")

    def infer(self, inputs):
        # Parse input data
        input_data = json.loads(inputs['input'][0])
        query = input_data['query']
        documents = input_data['documents']
        
        # Prepare the data for the model
        encoded_input = self.tokenizer([query]*len(documents), documents, return_tensors='pt', padding=True, truncation=True)

        # Run the inference
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        
        # Process the results
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1].tolist()
        results = [{"text": doc, "score": score} for doc, score in zip(documents, scores)]

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return {"output": json.dumps(results)}

    def finalize(self):
        self.model = None
        self.tokenizer = None
