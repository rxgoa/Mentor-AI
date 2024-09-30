import dspy
from uuid import UUID, uuid4
from llm.core.signatures.knowledge_signature import KnowledgeMasterOutput, Entry, StatusEntry

handcrafted_examples = [

    # User Attributes (5 examples)
    dspy.Example(
        input_text="I'm an introvert and prefer quiet environments",
        existing_knowledge=[],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="UserAttribute",
            content="Introvert, prefers quiet environments",
            status=StatusEntry("CreateMemory"),
            details=[{"reason": "personality trait"}, {"reason": "work preference"}],
            original_entry=Entry(id=None, category=None, content=None, details=None, status=None)
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I'm allergic to peanuts and shellfish",
        existing_knowledge=[],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="UserAttribute",
            content="Allergic to peanuts and shellfish",
            status=StatusEntry("CreateMemory"),
            details=[{"reason": "health information"}, {"reason": "dietary restriction"}],
            original_entry=Entry(id=None, category=None, content=None, details=None, status=None)
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I'm a night owl and most productive after 10 PM",
        existing_knowledge=[],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="UserAttribute",
            content="Night owl, most productive after 10 PM",
            status=StatusEntry("CreateMemory"),
            details=[{"reason": "work habit"}, {"reason": "personal rhythm"}],
            original_entry=Entry(id=None, category=None, content=None, details=None, status=None)
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I'm a visual learner and prefer diagrams and charts",
        existing_knowledge=[],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="UserAttribute",
            content="Visual learner, prefers diagrams and charts",
            status=StatusEntry("CreateMemory"),
            details=[{"reason": "learning style"}, {"reason": "information processing preference"}],
            original_entry=Entry(id=None, category=None, content=None, details=None, status=None)
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I'm passionate about environmental conservation",
        existing_knowledge=[],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="UserAttribute",
            content="Passionate about environmental conservation",
            status=StatusEntry("CreateMemory"),
            details=[{"reason": "personal value"}, {"reason": "lifestyle influence"}],
            original_entry=Entry(id=None, category=None, content=None, details=None, status=None)
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    #ShortTermGoals - Modifications
    dspy.Example(
        input_text="I've decided to read 'War and Peace' instead of 'The Alchemist' this week",
        existing_knowledge=[
           Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174000"),
                category="ShortTermGoals",
                content="Finish reading 'The Alchemist' by next week",
                details=[{"reason": "personal growth"}, {"reason": "expand literary knowledge"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="ShortTermGoals",
            content="Read 'War and Peace' this week",
            details=[{"reason": "challenge myself"}, {"reason": "explore classic literature"}],
            status=StatusEntry("UpdateMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174000"),
                category="ShortTermGoals",
                content="Finish reading 'The Alchemist' by next week",
                details=[{"reason": "personal growth"}, {"reason": "expand literary knowledge"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I need to reorganize my home office instead of my closet this weekend",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174001"),
                category="ShortTermGoals",
                content="Clean out closet this weekend",
                details=[{"reason": "organize living space"}, {"reason": "prepare for donation"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="ShortTermGoals",
            content="Reorganize home office this weekend",
            details=[{"reason": "improve work environment"}, {"reason": "increase productivity"}],
            status=StatusEntry("UpdateMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174001"),
                category="ShortTermGoals",
                content="Clean out closet this weekend",
                details=[{"reason": "organize living space"}, {"reason": "prepare for donation"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    # ShortTermGoals - Deletions
    dspy.Example(
        input_text="I've already finished 'The Alchemist', so I don't need to read it next week",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174002"),
                category="ShortTermGoals",
                content="Finish reading 'The Alchemist' by next week",
                details=[{"reason": "personal growth"}, {"reason": "expand literary knowledge"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="JournalEntry",
            content="Finished reading 'The Alchemist'",
            details=[{"reason": "completed ahead of schedule"}, {"reason": "enjoyed the book"}],
            status=StatusEntry("DeleteMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174002"),
                category="ShortTermGoals",
                content="Finish reading 'The Alchemist' by next week",
                details=[{"reason": "personal growth"}, {"reason": "expand literary knowledge"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I no longer need to clean out my closet as I did it yesterday",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174003"),
                category="ShortTermGoals",
                content="Clean out closet this weekend",
                details=[{"reason": "organize living space"}, {"reason": "prepare for donation"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="JournalEntry",
            content="Cleaned out closet yesterday",
            details=[{"reason": "task completed early"}, {"reason": "feeling organized"}],
            status=StatusEntry("DeleteMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174003"),
                category="ShortTermGoals",
                content="Clean out closet this weekend",
                details=[{"reason": "organize living space"}, {"reason": "prepare for donation"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    # MediumTermGoals - Modifications
    dspy.Example(
        input_text="I've decided to learn Italian instead of Spanish in the next 6 months",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174004"),
                category="MediumTermGoals",
                content="Learn basic Spanish in 6 months",
                details=[{"reason": "prepare for upcoming vacation"}, {"reason": "personal growth"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="MediumTermGoals",
            content="Learn basic Italian in 6 months",
            details=[{"reason": "change in travel plans"}, {"reason": "interest in Italian culture"}],
            status=StatusEntry("UpdateMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174004"),
                category="MediumTermGoals",
                content="Learn basic Spanish in 6 months",
                details=[{"reason": "prepare for upcoming vacation"}, {"reason": "personal growth"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I want to save $6000 for a new computer instead of $5000 by the end of the year",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174005"),
                category="MediumTermGoals",
                content="Save $5000 for a new computer by year-end",
                details=[{"reason": "upgrade work equipment"}, {"reason": "improve productivity"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="MediumTermGoals",
            content="Save $6000 for a new computer by year-end",
            details=[{"reason": "upgrade to better model"}, {"reason": "future-proof investment"}],
            status=StatusEntry("UpdateMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174005"),
                category="MediumTermGoals",
                content="Save $5000 for a new computer by year-end",
                details=[{"reason": "upgrade work equipment"}, {"reason": "improve productivity"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    # MediumTermGoals - Deletions
    dspy.Example(
        input_text="I've decided not to learn Spanish anymore as my travel plans have changed",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174006"),
                category="MediumTermGoals",
                content="Learn basic Spanish in 6 months",
                details=[{"reason": "prepare for upcoming vacation"}, {"reason": "personal growth"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="JournalEntry",
            content="Decided not to learn Spanish due to change in travel plans",
            details=[{"reason": "change in priorities"}, {"reason": "focus on other goals"}],
            status=StatusEntry("DeleteMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174006"),
                category="MediumTermGoals",
                content="Learn basic Spanish in 6 months",
                details=[{"reason": "prepare for upcoming vacation"}, {"reason": "personal growth"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I no longer need to save for a new computer as I received one as a gift",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174007"),
                category="MediumTermGoals",
                content="Save $5000 for a new computer by year-end",
                details=[{"reason": "upgrade work equipment"}, {"reason": "improve productivity"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="JournalEntry",
            content="Received new computer as a gift, no need to save",
            details=[{"reason": "unexpected gift"}, {"reason": "goal achieved differently"}],
            status=StatusEntry("DeleteMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174007"),
                category="MediumTermGoals",
                content="Save $5000 for a new computer by year-end",
                details=[{"reason": "upgrade work equipment"}, {"reason": "improve productivity"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    # LongTermGoals - Modifications
    dspy.Example(
        input_text="I want to start my own non-profit organization instead of a business in the next 5 years",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174008"),
                category="LongTermGoals",
                content="Start own business within 5 years",
                details=[{"reason": "financial independence"}, {"reason": "pursue passion"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="LongTermGoals",
            content="Start own non-profit organization within 5 years",
            details=[{"reason": "make a social impact"}, {"reason": "align work with values"}],
            status=StatusEntry("UpdateMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174008"),
                category="LongTermGoals",
                content="Start own business within 5 years",
                to_remove=[{"reason": "financial independence"}, {"reason": "pursue passion"}],
                details=[{"reason": "financial independence"}, {"reason": "pursue passion"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I now want to buy a house in the city instead of the countryside within 10 years",
        existing_knowledge=[
           Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174009"),
                category="LongTermGoals",
                content="Buy a house in the countryside within 10 years",
                details=[{"reason": "change lifestyle"}, {"reason": "investment in property"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="LongTermGoals",
            content="Buy a house in the city within 10 years",
            details=[{"reason": "prefer urban lifestyle"}, {"reason": "closer to work opportunities"}],
            status=StatusEntry("UpdateMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174009"),
                category="LongTermGoals",
                content="Buy a house in the countryside within 10 years",
                to_remove=[{"reason": "change lifestyle"}, {"reason": "investment in property"}],
                details=[{"reason": "change lifestyle"}, {"reason": "investment in property"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    # LongTermGoals - Deletions
    dspy.Example(
        input_text="I've decided not to start my own business anymore and focus on my current career",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174010"),
                category="LongTermGoals",
                content="Start own business within 5 years",
                details=[{"reason": "financial independence"}, {"reason": "pursue passion"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="JournalEntry",
            content="Decided to focus on current career instead of starting a business",
            details=[{"reason": "reevaluated priorities"}, {"reason": "found fulfillment in current job"}],
            status=StatusEntry("DeleteMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174010"),
                category="LongTermGoals",
                content="Start own business within 5 years",
                to_remove=[{"reason": "financial independence"}, {"reason": "pursue passion"}],
                details=[{"reason": "financial independence"}, {"reason": "pursue passion"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

    dspy.Example(
        input_text="I no longer want to buy a house as I prefer the flexibility of renting",
        existing_knowledge=[
            Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174011"),
                category="LongTermGoals",
                content="Buy a house in the countryside within 10 years",
                details=[{"reason": "change lifestyle"}, {"reason": "investment in property"}],
                status=StatusEntry("CreateMemory")
            )
        ],
        output=KnowledgeMasterOutput(
            id=uuid4(),
            category="JournalEntry",
            content="Decided against buying a house, prefer renting for flexibility",
            details=[{"reason": "change in lifestyle preference"}, {"reason": "value flexibility over ownership"}],
            status=StatusEntry("DeleteMemory"),
            original_entry=Entry(
                id=UUID("123e4567-e89b-12d3-a456-426614174011"),
                category="LongTermGoals",
                content="Buy a house in the countryside within 10 years",
                to_remove=[{"reason": "change lifestyle"}, {"reason": "investment in property"}],
                details=[{"reason": "change lifestyle"}, {"reason": "investment in property"}],
                status=StatusEntry("CreateMemory")
            )
        )
    ).with_inputs('input_text', 'existing_knowledge'),

]