import os
import dspy
from dspy.teleprompt import BootstrapFewShot
import json
import random
import threading
import time
from tqdm import tqdm

class SyntheticMessageGenerator(dspy.Signature):
    """Generate a synthetic user message for personal development, goals or life aspirations."""
    category = dspy.InputField(desc="Category: 'goal', 'task', 'journal', 'attribute', or 'irrelevant'")
    context = dspy.InputField(desc="Additional context for message generation")
    message = dspy.OutputField(desc="A synthetic user message")

def generate_synthetic_data(num_samples=50):
    generator = dspy.Predict(SyntheticMessageGenerator)
    synthetic_data = []
    categories = ['goal', 'task', 'journal', 'attribute', 'irrelevant']
    contexts = [
        "career advancement", "learning new skills", "personal finance",
        "health and fitness", "relationships", "daily activities",
        "material possessions"
    ]

    for _ in tqdm(range(num_samples), desc="Generating synthetic data"):
        category = random.choice(categories)
        context = random.choice(contexts)
        result = generator(category=category, context=context)
        relevance = "TRUE" if category != 'irrelevant' else "FALSE"
        example = dspy.Example(
            input_text=result.message,
            relevance=relevance,
            category=category,
            context=context
        ).with_inputs('input_text')
        synthetic_data.append(example)

    return synthetic_data

def validate_relevance(example, pred, trace=None):
    return example.relevance == pred.relevance

class CustomBootstrapFewShot(BootstrapFewShot):
    def __init__(self, metric):
        super().__init__(metric)
        self.successful_examples = []
        self.failed_examples = []
        self.bootstrapped_examples = []

    def compile(self, module, trainset, valset=None, max_rounds=3, max_traces=10, **config):
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
        input_example = dspy.Example(input_text=input_text).with_inputs('input_text')
        pred = module(input_example)
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

def initial_training(relevance_module=any):
    print("Generating synthetic data...")
    synthetic_trainset = generate_synthetic_data(num_samples=50)
    print(f"Generated {len(synthetic_trainset)} synthetic examples")

    original_trainset = [
        dspy.Example(input_text="today i was feeling pretty low energy but after taking a nap my body is back again", category="journal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="yesterday i saw a friend of mine that i havent seen for so long", category="journal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I feel really tired today and can't focus on my tasks.", category="task", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="Had a meeting with my friend.", category="irrelevant", relevance="FALSE").with_inputs('input_text'),
        dspy.Example(input_text="Today I worked on my project and made great progress.", category="task", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I'm excited to start learning Python next week.", category="goal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="Just finished watching a movie.", category="irrelevant", relevance="FALSE").with_inputs('input_text'),
        dspy.Example(input_text="I want to buy a car soon.", category="goal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I want to buy a Tesla.", category="goal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I am saving for a new laptop.", category="goal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I'm planning a family vacation.", category="goal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I'm saving for a house.", category="goal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="Had dinner with a friend.", category="irrelevant", relevance="FALSE").with_inputs('input_text'),
        dspy.Example(input_text="I want to improve my communication skills.", category="goal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I'm feeling stressed about my upcoming presentation.", category="journal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I need to buy groceries later.", category="irrelevant", relevance="FALSE").with_inputs('input_text'),
        dspy.Example(input_text="I'm considering changing my career path.", category="journal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="The weather is nice today.", category="irrelevant", relevance="FALSE").with_inputs('input_text'),
        dspy.Example(input_text="I saw a cute cat video on YouTube.", category="irrelevant", relevance="FALSE").with_inputs('input_text'),
        dspy.Example(input_text="I'm working on improving my time management skills.", category="journal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I'm feeling overwhelmed with my workload.", category="journal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I'm going to the gym to stay fit.", category="journal", relevance="TRUE").with_inputs('input_text'),
        dspy.Example(input_text="I'm reading a book on personal finance.", category="journal", relevance="TRUE").with_inputs('input_text')
    ]
    combined_trainset = original_trainset + synthetic_trainset
    print(f"Combined dataset size: {len(combined_trainset)}")

    print("Starting model compilation...")
    teleprompter = CustomBootstrapFewShot(metric=validate_relevance)
    compiled_relevance_detector = teleprompter.compile(relevance_module(), trainset=combined_trainset, max_rounds=5, max_traces=30)

    save_model_state(compiled_relevance_detector, combined_trainset)
    return compiled_relevance_detector

def save_model_state(compiled_model, trainset, filepath="relevance_classifier_state.json"):
    state = {
        "model_class": compiled_model.__class__.__name__,
        "trainset": [
            {"input_text": ex.input_text, "category": ex.category, "relevance": ex.relevance}
            for ex in trainset
        ]
    }
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"Model state saved to {filepath}")

def load_or_train_model(relevance_module=any):
    if os.path.exists("relevance_classifier_state.json"):
        print("\n\nLoading existing model state...\n\n")
        return load_model_state(relevance_module=relevance_module)
    else:
        print("No existing model state found. Performing initial training...")
        compiled_model = initial_training(relevance_module=relevance_module)
        _, trainset = load_model_state(relevance_module=relevance_module)  # Load the trainset after saving
        return compiled_model, trainset

def load_model_state(filepath="relevance_classifier_state.json", relevance_module=any):
    with open(filepath, 'r') as f:
        state = json.load(f)

    loaded_trainset = [
        dspy.Example(input_text=ex['input_text'], category=ex['category'], relevance=ex['relevance']).with_inputs('input_text', 'category')
        for ex in state['trainset']
    ]

    teleprompter = CustomBootstrapFewShot(metric=validate_relevance)
    compiled_model = teleprompter.compile(relevance_module(), trainset=loaded_trainset)

    return compiled_model, loaded_trainset

class RelevanceDetector:
    def __init__(self, RelevanceModule):
        self.model, self.trainset = load_or_train_model(relevance_module=RelevanceModule)
        self.lock = threading.Lock()
        self.valset = self.create_validation_set()

    def classify(self, input_text):
        with self.lock:
            example = dspy.Example(input_text=input_text).with_inputs('input_text')
            prediction = self.model(example)
            return prediction.relevance, prediction.explanation

    def retrain(self):
        new_model, new_trainset = load_or_train_model()
        with self.lock:
            self.model = new_model
            self.trainset = new_trainset
        self.evaluate()

    def evaluate(self):
        teleprompter = CustomBootstrapFewShot(metric=validate_relevance)
        accuracy = teleprompter.evaluate(self.model, self.valset)
        print(f"Current model accuracy: {accuracy:.2f}")

    def create_validation_set(self):
        # Create a small validation set
        return generate_synthetic_data(num_samples=50)

    def print_trainset(self):
        print("Current trainset:")
        for idx, example in enumerate(self.trainset):
            print(f"{idx + 1}. Input: {example.input_text}")
            print(f"   Category: {example.category}")
            print(f"   Relevance: {example.relevance}")
            print()

def periodic_retraining(detector, interval=86400):  # 86400 seconds = 1 day
    while True:
        time.sleep(interval)
        print("Retraining model...")
        detector.retrain()
        print("Model retrained successfully.")


# Start periodic retraining in a separate thread
# retraining_thread = threading.Thread(target=periodic_retraining, args=(detector,))
# retraining_thread.daemon = True
# retraining_thread.start()