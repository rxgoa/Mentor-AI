import os
import dspy
from dspy.teleprompt import BootstrapFewShot
import json
import random
from tqdm import tqdm
from functools import lru_cache
from uuid import uuid4
from llm.core.signatures.knowledge_signature import KnowledgeMasterOutput, Entry, StatusEntry
from llm.core.trainer.handcraft_examples import handcrafted_examples


class SyntheticMessageGenerator(dspy.Signature):
    """Generate a synthetic user message related to personal development, goals, or life aspirations."""
    category = dspy.InputField(desc="Category: 'ShortTermGoals', 'MediumTermGoals', 'LongTermGoals', 'Task', 'JournalEntry', 'UserAttribute'")
    context = dspy.InputField(desc="Additional context for message generation")
    message = dspy.OutputField(desc="A synthetic user message")


# TODO:
# i need to create better synthetic data generation
# update entry scenarios -> status and input_text should reflect the intent (updating details or content)
# delete entry scenarios -> status and input_text should reflect the intent
# delete detail entry scenarios -> status and input_text should reflect the intent

@lru_cache(maxsize=1)
def generate_synthetic_data(num_samples=10):
    generator = dspy.Predict(SyntheticMessageGenerator)
    synthetic_data = []
    categories = ['ShortTermGoals', 'MediumTermGoals', 'LongTermGoals', 'Task', 'JournalEntry', 'UserAttribute']
    categories_weights = [1, 1, 1, 1, 1, 1] # for even distribution
    contexts = [
        "career advancement", "learning new skills", "personal finance",
        "health and fitness", "relationships", "daily activities",
        "material possessions", "mental well-being", "travel and exploration", "community involvement",
        "hobbies and interests", "spirituality", "work-life balance"
    ]

    for _ in tqdm(range(num_samples), desc="Generating synthetic data"):
        new_id = uuid4()
        category = random.choices(categories, weights=categories_weights, k=1)[0]
        context = random.choice(contexts)
        result = generator(category=category, context=context)
        example = dspy.Example(
            input_text=result.message,
            existing_knowledge=[Entry()],  # Empty list for simplicity
            output=KnowledgeMasterOutput(
                id=new_id,
                category=category,
                content=result.message,
                details=[{"reason": context}],
                status=StatusEntry("CreateMemory"),
                original_entry=Entry(id=None, category=None, content=None, details=None, status=None, to_remove=[])
            )
        ).with_inputs('input_text', 'existing_knowledge')
        synthetic_data.append(example)

    return synthetic_data

def validate_knowledge_master(example, pred, trace=None):
    if pred is None or example is None:
        return False

    correct_category = example.output.category == pred["output"].category
    correct_content = example.output.content.lower() in pred["output"].content.lower()
    return correct_category and correct_content

class CustomBootstrapFewShot(BootstrapFewShot):
    def __init__(self, metric):
        super().__init__(metric)
        self.successful_examples = []
        self.failed_examples = []
        self.bootstrapped_examples = []

    def compile(self, module, trainset, valset=None, max_rounds=1, max_traces=1, **config):
        print(f"Starting compilation with {len(trainset)} total examples")
        print(f"Max rounds: {max_rounds}, Max traces per round: {max_traces}")

        self.bootstrapped_examples = list(trainset)
        random.shuffle(self.bootstrapped_examples)  # Shuffle for better learning

        for round in range(max_rounds):
            print(f"Starting round {round + 1}")
            traces_this_round = 0
            for _ in tqdm(range(max_traces), desc=f"Round {round + 1} traces"):
                if not self.bootstrapped_examples:
                    break
                self.step(module)
                traces_this_round += 1

            print(f"Completed {traces_this_round} traces in round {round + 1}")
            if traces_this_round == 0:
                break

        # Evaluate on valset if provided
        if valset:
            accuracy = self.evaluate(module, valset)
            print(f"Validation accuracy: {accuracy:.2f}")

        return module

    def step(self, module):
        if self.bootstrapped_examples:
            example = self.bootstrapped_examples.pop(0)
            self._bootstrap_one_example(module, example)
        else:
            print("No more examples to bootstrap.")

    def _bootstrap_one_example(self, module, example):
        input_text = example.input_text
        existing_knowledge = example.existing_knowledge

        pred = module(input_text, existing_knowledge)

        if self.metric(example, pred):
            self.successful_examples.append(example)
        else:
            self.failed_examples.append(example)

    def evaluate(self, module, valset):
        correct = 0
        for example in valset:
            pred = module(example)
            if self.metric(example, pred):
                correct += 1
        return correct / len(valset)

def initial_training(knowledge_master_module):
    print("Generating synthetic data...")
    synthetic_trainset = generate_synthetic_data(num_samples=10)
    print(f"Generated {len(synthetic_trainset)} synthetic examples")

    loaded_examples = handcrafted_examples

    combined_trainset = loaded_examples + synthetic_trainset
    print(f"Combined dataset size: {len(combined_trainset)}")

    print("Starting model compilation...")
    teleprompter = CustomBootstrapFewShot(metric=validate_knowledge_master)
    compiled_knowledge_master = teleprompter.compile(knowledge_master_module(), trainset=combined_trainset, max_rounds=2, max_traces=5)
    save_model_state(compiled_knowledge_master, combined_trainset)
    return compiled_knowledge_master

def save_model_state(compiled_model, trainset, filepath="knowledge_master_state.json"):
    state = {
        "model_class": compiled_model.__class__.__name__,
        "trainset": [
            {
                "input_text": ex.input_text,
                "existing_knowledge": [{
                    "id": str(ex.existing_knowledge[0].id),
                    "category":ex.existing_knowledge[0].category,
                    "content":ex.existing_knowledge[0].content,
                    "details":ex.existing_knowledge[0].details,
                    "status":ex.existing_knowledge[0].status
                }] if len(ex.existing_knowledge) > 0 else [],
                "output": {
                    "id": str(ex.output.id),
                    "category": ex.output.category,
                    "content": ex.output.content,
                    "details": [{
                        "reason": detail.reason,
                    } for detail in ex.output.details ] if len(ex.output.details) > 0 else [],
                    "status": ex.output.status,
                    "original_entry": {
                                    "id": str(ex.output.original_entry.id),
                                    "category":ex.output.original_entry.category,
                                    "content":ex.output.original_entry.content,
                                    "details":ex.output.original_entry.details,
                                    "to_remove":ex.output.original_entry.to_remove,
                                    "status":ex.output.original_entry.status
                    } if ex.output.original_entry.id is not None else None
                }
            }
            for ex in trainset
        ]
    }
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"Model state saved to {filepath}")

def load_or_train_model(knowledge_master_module):
    if os.path.exists("knowledge_master_state.json"):
        print("\n\nLoading existing model state...\n\n")
        return load_model_state(knowledge_master_module=knowledge_master_module)
    else:
        print("No existing model state found. Performing initial training...")
        compiled_model = initial_training(knowledge_master_module=knowledge_master_module)
        _, trainset = load_model_state(knowledge_master_module=knowledge_master_module)  # Load the trainset after saving
        return compiled_model, trainset

@lru_cache(maxsize=1)
def load_model_state(filepath="knowledge_master_state.json", knowledge_master_module=None):
    with open(filepath, 'r') as f:
        state = json.load(f)

    print(f"\n\nLOADED TRAINSET LENGTH: {len(state['trainset'])}")

    loaded_trainset = [
        dspy.Example(
            input_text=ex['input_text'],
            existing_knowledge=ex['existing_knowledge'],
            output=KnowledgeMasterOutput(
                id=ex['output']['id'],
                category=ex['output']['category'],
                content=ex['output']['content'],
                details=ex['output']['details'],
                status=ex['output']['status'],
                original_entry=Entry(
                  id=ex['output']['original_entry']['id'],
                  category=ex['output']['original_entry']['category'],
                  content=ex['output']['original_entry']['content'],
                  details=ex['output']['original_entry']['details'],
                  to_remove=ex['output']['original_entry']['to_remove'],
                  status=StatusEntry(ex['output']['original_entry']['status']),
                ) if ex['output']['original_entry'] is not None else None
            )
        ).with_inputs('input_text', 'existing_knowledge')
        for ex in state['trainset']
    ]

    teleprompter = CustomBootstrapFewShot(metric=validate_knowledge_master)
    compiled_model = teleprompter.compile(knowledge_master_module(), trainset=loaded_trainset)

    return compiled_model, loaded_trainset

class KnowledgeMasterTrainer:
    def __init__(self, KnowledgeMasterModule):
        self.model, self.trainset = load_or_train_model(knowledge_master_module=KnowledgeMasterModule)
        self.valset = self.create_validation_set()

    def process(self, input_text, existing_knowledge):
        prediction = self.model(input_text=input_text, existing_knowledge=existing_knowledge)
        return prediction["output"]

    def retrain(self):
        new_model, new_trainset = load_or_train_model(knowledge_master_module=self.model.__class__)
        self.model = new_model
        self.trainset = new_trainset
        self.evaluate()

    def evaluate(self):
        teleprompter = CustomBootstrapFewShot(metric=validate_knowledge_master)
        accuracy = teleprompter.evaluate(self.model, self.valset)
        print(f"Current model accuracy: {accuracy:.2f}")

    def create_validation_set(self):
        return generate_synthetic_data(num_samples=10)

    def print_trainset(self):
        print("Current trainset:")
        for idx, example in enumerate(self.trainset):
            print(f"{idx + 1}. Input: {example.input_text}")
            print(f"   Existing Knowledge: {example.existing_knowledge}")
            print(f"   Output: {example.output}")
            print()

# Uncomment this piece of code when training. For now we do this, after when we schedlue a retrain, we should rethink how to approach this.
# from llm.core.modules.knowledge_module import KnowledgeMaster
# GROQ_API_KEY = os.environ['GROQ_API_KEY']
# groq = dspy.LM('groq/llama3-70b-8192', api_key=GROQ_API_KEY)
# dspy.settings.configure(lm=groq)
# trainer = KnowledgeMasterTrainer(KnowledgeMaster)
# result = trainer.process("I want to learn Python", [])
# print(result)
