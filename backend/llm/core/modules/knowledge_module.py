import dspy
from pprint import pprint
import json
import re
from llm.core.signatures.knowledge_signature import KnowledgeMasterOutput, OutputDetails, Entry, AnalyzeInput, ExtractDetails, CheckOriginalEntry, MergeKnowledge, FinalizeOutput
from uuid import uuid4, UUID

class KnowledgeMaster(dspy.Module):

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnalyzeInput)#
        self.extract_details = dspy.ChainOfThought(ExtractDetails)#
        self.check_original = dspy.ChainOfThought(CheckOriginalEntry)
        self.merge = dspy.ChainOfThought(MergeKnowledge)
        self.finalize = dspy.ChainOfThought(FinalizeOutput)

    def forward(self, input_text, existing_knowledge):
        try:
            instruc = "Create a new Entry"
            merged = {}

            analyzed_input = self.analyze(input_text=input_text)
            input_details = self.extract_details(input_text=input_text, category=analyzed_input["category"], content=analyzed_input["content"], status=analyzed_input.status.value)
            ex_knowledge = self.check_existing_knowledge(input_text=input_text, input_analysis=analyzed_input.reasoning, existing_knowledge=existing_knowledge)
            cur_entry = self.combined_entry(analyzed_input, input_details)

            if len(ex_knowledge) > 0:
                # we need to compare current entry to existing knowledge to find the entry to be deleted or updated.
                merged = self.merge(old_knowledge=ex_knowledge, new_knowledge=cur_entry)
                instruc = merged.reasoning

            finalize = self.format_finalize(instructions_to_follow=instruc, combined_new_entry=cur_entry, existing_knowledge_match=merged)
            final_response = self.final_response(finalize, merged)

            return {"output": final_response, "context": { "ex_knowledge": ex_knowledge, "compared": merged, "analyzed_input": analyzed_input }}

        except Exception as e:
            pprint(e)

    def final_response(self, finalize, merged):
        return KnowledgeMasterOutput(
                id=finalize["id"],
                category=finalize["category"],
                content=finalize["content"],
                details=finalize["details"],
                status=finalize["status"],
                context_entry=merged,
                original_entry=finalize["original_entry"] if "original_entry" in finalize else {}
            )


    def combined_entry(self, analyzed_input, input_details):
         return Entry(
                id=uuid4(),
                category=analyzed_input["category"],
                content=analyzed_input["content"],
                status=analyzed_input.status,
                details=[{
                    "reason": detail.reason
                } for detail in input_details.details] if len(input_details.details) > 0 else []
            )

    def format_finalize(self, instructions_to_follow, combined_new_entry, existing_knowledge_match):
        try:
            return self.finalize(instructions_to_follow=instructions_to_follow, combined_new_entry=combined_new_entry, existing_knowledge_match=existing_knowledge_match)
        except Exception as e:
            pprint(e)

    def check_existing_knowledge(self, input_text, input_analysis, existing_knowledge):
        ex_knowledge = []
        # this is a temp measure to make sure we have a valid existing_knowledge
        if len(existing_knowledge) > 0 and existing_knowledge[0]["category"] is not None:
            entry_obj = [
                {
                    "id": entry["id"],
                    "category": entry["category"],
                    "content": entry["content"],
                    "status": entry["status"],
                    "details": [
                        {
                            "reason": detail.reason if isinstance(entry["details"][0], OutputDetails) else detail["reason"]
                        } for detail in entry["details"]] if len(entry["details"]) > 0 else []
                } for entry in existing_knowledge]

            knowledge = self.check_original(input_text=input_text, input_analysis=input_analysis, compare_existing_entries_with_input_text_intent=entry_obj)

            # TODO: for now we do this hacky thing
            if isinstance(knowledge.match_entry, str):
                not_back = re.sub(r"\'", '"', knowledge.match_entry)
                formatted = not_back.replace("'", '"')
                json_loaded = json.loads(formatted)
                if isinstance(json_loaded, dict):
                    ex_knowledge = [json_loaded]
                else:
                    ex_knowledge = json_loaded

        return ex_knowledge