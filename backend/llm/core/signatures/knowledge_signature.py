import dspy
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

class StatusEntry(str, Enum):
    CreateMemory = "CreateMemory"
    DeleteMemory = "DeleteMemory"
    UpdateMemory = "UpdateMemory"
    DeleteMemoryDetail = "DeleteMemoryDetail"

class Entry(BaseModel):
    id: Optional[UUID] = Field(default_factory=uuid4)
    category: Optional[str] = None
    content: Optional[str] = None
    details: Optional[List[Dict[str, str]]] = None
    status: Optional[StatusEntry] = None
    to_remove: Optional[List[Dict[str, str]]] = [] # list of details to be removed.

class OutputDetails(BaseModel):
    reason: str

class KnowledgeMasterOutput(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    category: str
    content: str
    status: StatusEntry
    details: List[OutputDetails] = []
    original_entry: Optional[Union[Any, Entry]] = {}


class KnowledgeMasterSignature(dspy.Signature):
    """
    Analyze input: Extract & categorize info. Handle updates & details updates if present.
    Categories: ShortTermGoals, MediumTermGoals, LongTermGoals, Task, JournalEntry, UserAttribute.
    Output: category, content, details (list of reasons, key='reason', value=explicit user reason), status (CreateMemory, DeleteMemory, UpdateMemory, DeleteMemoryDetail), original_entry (if exists, else {}).
    Updates/detail deletion: Compare with existing. Insert details to remove in original_entry['to_remove'].
    Preserve original details, add new ones for evolving preferences.
    """
    input_text: str = dspy.InputField(desc="The most recent message from the user")
    existing_knowledge: List[Dict[str, str]] = dspy.InputField(desc="List of existing knowledge entries.")
    output: KnowledgeMasterOutput = dspy.OutputField(desc="Output formatted of the relevant input, including explicit update instructions.")

class AnalyzeInput(dspy.Signature):
    """Extract and categorize high-level information from the input. Respect attributes descriptions if present. You identify user's intent for creating, altering or deletion.
    Don't read too much in the user's wording. Extract what they meant in a clear way. As if you were going to explain to a 12 year old."""
    input_text: str = dspy.InputField()
    category: str = dspy.OutputField(desc="Categories for the given input. Only valid options are: ShortTermGoals, MediumTermGoals, LongTermGoals, Task, JournalEntry and UserAttribute")
    content: str = dspy.OutputField()
    status: StatusEntry = dspy.OutputField(desc="Status for the given input. Only valid options are: CreateMemory, UpdateMemory, DeleteMemory and DeleteMemoryDetail")

class ExtractDetails(dspy.Signature):
    """Extract a list of details with concise information and reasons from the analyzed input.
    Each detail must be a dictionary with a single key-value pair where the key is always 'reason'."""
    input_text: str = dspy.InputField()
    category: str = dspy.InputField()
    content: str = dspy.InputField()
    details: List[OutputDetails] = dspy.OutputField(desc="List of details with reasons extracted from the input")

class CheckOriginalEntry(dspy.Signature):
    """
        Analyze the input_text and input_analysis to understand the user's intent.
        Compare this intent with each entry in compare_existing_entries_with_input_text_intent.
        Identify any entries that the new input might be referencing, updating, or relating to.
        Consider semantic relationships, not just exact matches or categories.
        If multiple entries seem relevant, include all of them.
        In your output, you should only insert entries from compare_existing_entries_with_input_text_intent.
        You should never alter values inside compare_existing_entries_with_input_text_intent. Only return the raw data filtered.
        Inside the json key "to_remove" should always be an empty array [].
        Explain your reasoning for each matched entry.
    """
    compare_existing_entries_with_input_text_intent: List[Dict[str, Any]] = dspy.InputField(desc="List of Entries. Take this list in context with the user's intent. If the user's intent matchs any Entry, you should identify.")
    input_analysis: str = dspy.InputField(desc="Analysis of the input_text attribute. The input_analysis refeers to if the user's intent is to alter, delete or create a entry.")
    input_text: str = dspy.InputField(desc="Input text from the user with the intent. We should check if any existing_entry match what the user wants")
    match_entry = dspy.OutputField(desc=f"The answer written as a list of JSON object using the schema {Entry()}. Valid status: CreateMemory, UpdateMemory, DeleteMemory and DeleteMemoryDetail")

class FinalizeOutput(dspy.Signature):
    """
    instructions_to_follow, combined_new_entry, existing_knowledge_match -> output.
    Never create new attributes that aren't explicitly in the signature of this module.
    For DeleteMemoryDetail, ensure that the specific detail is added to the 'to_remove' list and removed from the 'details' list.
    For DeleteMemory, set the status accordingly without modifying the content.
    """
    instructions_to_follow: str = dspy.InputField(desc="Instructions to follow given the user intent. Use only your available from inputs to achieve the result wanted.")
    combined_new_entry: Entry = dspy.InputField(desc="Current Entry being processed.")
    existing_knowledge_match: Optional[List[Entry]] = dspy.InputField(desc="List of existing Entries where the current Entry context matchs.")
    id: str = dspy.OutputField(desc="Valid UUID")
    category: str = dspy.OutputField(desc="Category of the output")
    content: str = dspy.OutputField(desc="Content of the output")
    details: List[Dict[str, str]] = dspy.OutputField(desc="List of details with reasons")
    status: str = dspy.OutputField(desc="Status of the output (CreateMemory, UpdateMemory, DeleteMemory, or DeleteMemoryDetail)")
    to_remove: List[Dict[str, Any]] = dspy.OutputField(desc="List of items to be removed")


class MergeKnowledge(dspy.Signature):
    """
    Compare old knowledge with new knowledge and produce a merged, updated, or deleted entry.
    Determine the appropriate status based on the user's intent:
    - DeleteMemory: When the user explicitly states they no longer want or need something previously stored.
    - UpdateMemory: When the user wants to add or modify information.
    - DeleteMemoryDetail: When the user wants to remove a specific detail without providing new detail to be added.

    Instructions:
    1. Carefully analyze the new_knowledge to understand the user's intent.
    2. If the user is adding or modifying, use UpdateMemory status and merge the old and new knowledge.
    3. If the user explicitly states they no longer want or need something previously stored, use DeleteMemory status.
       Do not update the content; instead, mark it for deletion.
    4. If the user is expressing a desire to delete a specific detail of knowledge, use DeleteMemoryDetail status.
    5. When updating, merge both old_knowledge and new_knowledge content, writing as if you were the user.
    6. For DeleteMemoryDetail, identify the specific detail to remove and add it to the 'to_remove' list.
    7. Always include the determined status within the modify_knowledge output.

    Remember:
    - The status should accurately reflect the user's intent to update, delete entirely, or delete a specific detail.
    - For DeleteMemory, do not modify the content. Simply set the status and leave the content as is.
    """
    old_knowledge: List[Dict[str, Any]] = dspy.InputField(desc="Existing knowledge about the user's desires or plans")
    new_knowledge: Entry = dspy.InputField(desc="New information or updates about the user's desires or plans")

    modify_knowledge: Dict[str, Any] = dspy.OutputField(desc="Merged and updated knowledge entry, incorporating new information into the existing structure")
