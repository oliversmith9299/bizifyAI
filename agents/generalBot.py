"""
✅ Allowed:

Read agent_runs

Read ideas

Read session profile

Summarize

Explain

Compare

Clarify

Answer “why / how / what does this mean”

❌ Forbidden:

Making new recommendations

Changing stored outputs

Running other agents

Advancing the business flow


User Question
   ↓
ProjectKnowledgeAgent
   ↓
Fetch agent_runs WHERE idea_id = ?
   ↓
Build temporary context (RAG-style)
   ↓
LLM answers using ONLY stored data


"""