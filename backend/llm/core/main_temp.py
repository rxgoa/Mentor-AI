import os
from pprint import pprint
import litellm
import dspy
from llm.core.trainer.trainer_relevance_input import RelevanceDetector
from llm.core.trainer.trainer_knowledge_master import KnowledgeMasterTrainer
from llm.core.modules.relevance_module import RelevanceModule
from llm.core.modules.knowledge_module import KnowledgeMaster
from llm.core.signatures.knowledge_signature import KnowledgeMasterOutput, StatusEntry
from typing import List, Dict
from langgraph.graph import StateGraph, END, START

#litellm.drop_params = True
GROQ_API_KEY = os.environ['GROQ_API_KEY']
groq = dspy.LM('groq/llama3-70b-8192', api_key=GROQ_API_KEY, max_tokens=500)
dspy.settings.configure(lm=groq)

class State(Dict):
    messages: List[str]
    memory: List[Dict[str, str]]
    analyzer_response: any
    relevance: str
    input_text: str
    knowledge_master_response: KnowledgeMasterOutput

def input_analyzer_node(state: State):
    input_text = state['input_text']
    messages = state["messages"] + [f"User: {input_text}"]
    input_text = state.get('input_text', '')
    detector = RelevanceDetector(RelevanceModule)
    result = detector.model(dspy.Example(input_text=input_text).with_inputs('input_text'))
    messages = state["messages"].append(f"User: {input_text}")
    return {"analyzer_response":  {
            "relevance": "TRUE" in result.relevance and "yes" or "no",
            "explanation": result.explanation,
            "thoughts": result.thoughts,
            "category": result.category
        }, "input_text": input_text, "messages": messages, "relevance": "TRUE" in result.relevance and "yes" or "no"}

def knowledge_master_node(state: State):
    input_text = state["input_text"]
    existing_knowledge = state['memory']
    detector = KnowledgeMasterTrainer(KnowledgeMaster)

    km_result = detector.model(input_text=input_text, existing_knowledge=existing_knowledge)
    return {"knowledge_master_response": km_result}

def knowledge_modifier_node(state: State):

    output = state["knowledge_master_response"]["output"]
    context = state["knowledge_master_response"]["context"]
    print("output:")
    print(f"\n\n{output}\n\n")
    print("context:")
    print(f"\n\n{context}\n\n")
    memory= state["memory"]

    if StatusEntry(output.status) == StatusEntry("CreateMemory"):
        # new memory
        print(f"\n\n\n NEW MEMORY \n\n")
        new_memory = {
            "id": str(output.id),
            "content": output.content,
            "category": output.category,
            "status": StatusEntry("CreateMemory"),
            "details": [],
            "original_entry": {}
        }

        new_memory["details"] = output.details

        state["memory"].append(new_memory)

    elif StatusEntry(output.status) == StatusEntry("UpdateMemory"):
        print(f"\n\n\n UPDATING MEMORY \n\n")
        original_entry = context["ex_knowledge"][0] # for now we do this.
        original_id = original_entry["id"]
        for mem in memory:
            if mem["id"] == original_id:
                mem["status"] = StatusEntry("UpdateMemory")
                # maybe change this later
                mem["content"] = output.content
                mem["category"] = output.category
                mem["details"] = []
                for detail in output.details:
                    mem["details"].append(detail)

    elif StatusEntry(output.status) == StatusEntry("DeleteMemory"):
        # check if we want to delete only details from a memory
        original_entry = context["ex_knowledge"][0] # for now we do this.
        print(f"\n DELETING MEMORY ID: {original_entry["id"]} \n")
        pprint(state["memory"])
        for i in state['memory']:
            if i['id'] == original_entry["id"]:
                print("\nMemory to be deleted:")
                pprint(i)

        updated_memory = [item for item in state["memory"] if item["id"] != original_entry["id"]]
        pprint(updated_memory)
        memory = updated_memory

    elif StatusEntry(output.status) == StatusEntry("DeleteMemoryDetail"):
        print(f"\n DELETING DETAILS OF MEMORY ID: {output.original_entry.id} \n")
        pprint(output)
    else:
        print(f"\n OTHER STATUS OF MEMORY ID: {output.original_entry.id} \n")

    state["memory"] = memory


    return state

def response_generator_node(state: State):#
    memory = state["memory"]
    messages = state['messages']
    final_response = "Input good!"
    if "no" in state["relevance"].lower():
        final_response = "Input not relevant!"

    return {"messages": messages, "final_response": final_response, "memory": memory}

workflow = StateGraph(State)
workflow.add_node("input_analyzer", input_analyzer_node)
workflow.add_node("knowledge_master", knowledge_master_node)
workflow.add_node("response_generator", response_generator_node)
workflow.add_node("knowledge_modifier", knowledge_modifier_node)

workflow.set_entry_point("input_analyzer")

workflow.add_conditional_edges(
    "input_analyzer",
    lambda x: x["relevance"],
    {
        "yes": "knowledge_master",
        "no": "response_generator",
    },
)
# workflow.add_edge("input_analyzer", "knowledge_master")
workflow.add_edge("knowledge_master", "knowledge_modifier")
workflow.add_edge("knowledge_modifier", "response_generator")
workflow.add_edge("response_generator", END)

initial_state = {
    "messages": [],
    "memory": [],
    "input_text": "",
}

def process_input(input_text: str):
    initial_state["input_text"] = input_text
    final_state = app.invoke(initial_state)
    return final_state

app = workflow.compile()

if __name__ == "__main__":
    while True:
        user_input = input("\nEnter your message (or 'quit' to exit): ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        state = process_input(user_input)

        print(f"\n\n final_state:")
        initial_state["memory"] = state["memory"]
        pprint(initial_state["memory"])
        pprint(state["messages"])
        print(f"Input quality: {state["relevance"]}")
