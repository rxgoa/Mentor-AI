import dspy

class RelevanceClassifier(dspy.Signature):
    """Classify whether an input is relevant to personal development, goals, or life aspirations.

    Relevance should be interpreted broadly, including:
    1. Explicit mentions of personal growth, skill development, or life goals.
    2. Implied personal objectives or aspirations, even if not directly stated as such.
    3. Actions or desires that could contribute to personal progress or life improvement.
    4. Financial goals or milestones that may impact one's life trajectory.
    5. Career-related intentions or plans.
    6. Health and wellness objectives.
    7. Relationship or social life goals.
    8. Material acquisitions that might signify personal progress or life changes.

    Consider the potential underlying motivations or implications of the statement,
    not just its surface-level content. If the statement could reasonably be part of
    a person's broader life plan or personal development journey, consider it relevant.
    """
    input_text = dspy.InputField()
    category = dspy.OutputField(desc="Category for each the relevant input is identified as: 'goal', 'task', 'journal', 'attribute', or 'irrelevant'")
    thoughts = dspy.OutputField(desc="Step-by-step reasoning about the relevance of the input, considering the broader context of personal development and life goals")
    relevance = dspy.OutputField(desc="TRUE if relevant to any aspect of personal development or life goals (interpreted broadly), FALSE otherwise")
    explanation = dspy.OutputField(desc="Brief explanation for the classification, including any inferred relevance to personal development or life goals")