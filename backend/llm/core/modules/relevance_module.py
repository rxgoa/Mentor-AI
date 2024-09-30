import dspy
from llm.core.signatures.relevance_signature import RelevanceClassifier

class RelevanceModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(RelevanceClassifier)

    def forward(self, input_text):
        classification = self.classifier(input_text=input_text.input_text)
        relevance = classification.relevance.strip().upper()
        explanation = classification.explanation
        thoughts = classification.thoughts
        category = classification.category

        if relevance not in ["TRUE", "FALSE"]:
            relevance = "FALSE"

        return dspy.Prediction(relevance=relevance, explanation=explanation, thoughts=thoughts, category=category)