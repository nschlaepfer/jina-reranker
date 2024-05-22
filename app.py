import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

class InferlessPythonModel:
    def initialize(self):
        # Load the model and tokenizer from Hugging Face with ignore_mismatched_sizes
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v1-tiny-en")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "jinaai/jina-reranker-v1-tiny-en",
            ignore_mismatched_sizes=True
        )

    def infer(self, inputs):
        try:
            # Parse input data
            input_data = json.loads(inputs['inputs'][0]['data'][0])
            query = input_data['query']
            documents = input_data['documents']
            
            # Prepare the data for the model
            encoded_input = self.tokenizer([query]*len(documents), documents, return_tensors='pt', padding=True, truncation=True)

            # Run the inference
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            
            # Extract scores and sort results
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1].tolist()
            results = [{"text": doc, "score": score} for doc, score in zip(documents, scores)]
            results.sort(key=lambda x: x['score'], reverse=True)

            return {"outputs": [{"name": "output", "data": [json.dumps(results)], "datatype": "BYTES"}]}
        except KeyError as e:
            return {"error": f"KeyError: {str(e)}"}

    def finalize(self):
        self.model = None
        self.tokenizer = None
