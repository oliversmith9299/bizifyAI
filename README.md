# bizifyAI

bizifyAI is an AI-assisted startup planning pipeline that helps founders move from a questionnaire or a raw idea into a validated, execution-ready business plan. The project is organized as a sequence of specialized agents. Each agent owns one stage of the startup-building process and produces structured output that can be saved, reviewed, refined, and passed to the next stage.

The system supports two main user scenarios:

- **New user flow:** the founder completes a questionnaire, then the system analyzes their profile, discovers realistic problems, and generates a personalized startup idea.
- **Returning user flow:** the founder already has an idea, so the system structures that idea, researches real problems around it, and helps refine it through chat.

After an idea is saved, the pipeline continues through customer analysis, competition, market potential, idea strategy, business model, product functions, MVP planning, unit economics, and go-to-market planning.

## Pipeline Sequence

```text
START
|
|-- New user:
|   OneProfileAnalysis.py
|   TwoProblemDiscovery.py
|   ThreePersonalizeIdeaChat.py
|
|-- Returning user with idea:
|   ThreeIdeaIntakeAgent.py
|   TwoProblemDiscovery.py
|   ThreePersonalizeIdeaChat.py
|
IDEA SAVED
|
FourCustomersAgent.py
FiveCompetitionAgent.py
SixMaketPotential.py
SevenIdeaStrategy.py
EightBusinessModel.py
NineFunctionsList.py
TenMVPPlanning.py
ElevenUnitEconomicsAgent.py
TwelveGoToMarket.py
```

## Agent Reference

### `OneProfileAnalysis.py`

Analyzes a new user's questionnaire output and builds a founder profile. It identifies the user's personality, motivation, strengths, weaknesses, skill gaps, risk level, readiness, preferred business direction, and realistic problem spaces. Its output is used by `TwoProblemDiscovery.py` to search for problems that match the founder's abilities, interests, region, and business goals.

Expected output file:

```text
data/profileAnalysis.json
```

### `TwoProblemDiscovery.py`

Researches real-world problems based on either the founder profile or a returning user's structured idea. It generates search queries, gathers external evidence, extracts pain points, identifies customer problems, scores validation strength, and returns problem opportunities that can become startup ideas. This agent makes sure the pipeline is grounded in real user needs instead of only assumptions.

Expected output file:

```text
data/problems.json
```

### `ThreeIdeaIntakeAgent.py`

Used only in the returning-user flow. This agent receives the user's raw idea and turns it into structured data. It extracts the idea summary, target users, industry, region, assumed problem, assumed solution, business model, and research keywords. If the idea is unclear, it should ask short clarification questions before sending anything to problem discovery.

Expected output file:

```text
data/IdeaIntake.json
```

### `ThreePersonalizeIdeaChat.py`

Generates and discusses the startup idea with the user. In the new-user flow, it creates the first personalized idea from the profile and discovered problems. In the returning-user flow, it refines the user's own idea using the structured idea intake and problem discovery results. This agent supports conversation, alternatives, refinements, and saving the selected idea.

Expected output files:

```text
data/idea_output.json
data/idea_chat_output.json
data/personalizedideachat.json
```

### `FourCustomersAgent.py`

Defines the target customers for the saved idea. It turns the selected idea and discovered problem into clear customer segments, identifies who feels the pain most, explains why they care, and describes how they behave. It can also use CATWOE analysis to understand customers, actors, transformation, worldview, owner, and environmental constraints.

Expected output file:

```text
data/FourCustomersAgent.json
```

### `FiveCompetitionAgent.py`

Analyzes the competitive landscape around the saved idea. It identifies direct competitors, indirect alternatives, substitute solutions, pricing models, feature differences, acquisition channels, and positioning gaps. It can also use Porter's Five Forces and VRIO analysis to evaluate competitive pressure and possible defensibility.

Expected output file:

```text
data/FiveCompetitionAgent.json
```

### `SixMaketPotential.py`

Estimates the market opportunity for the saved idea. It defines the market, target region, TAM, SAM, SOM, major trends, growth drivers, adoption barriers, and opportunity attractiveness. It can also use PESTEL analysis to evaluate political, economic, social, technological, environmental, and legal factors.

Expected output file:

```text
data/SixMaketPotential.json
```

### `SevenIdeaStrategy.py`

Turns the saved idea into a clear strategic direction. It defines the value proposition, positioning, core promise, differentiation strategy, key assumptions, validation priorities, and a high-level business plan. This agent acts as the bridge between research and execution.

Expected output file:

```text
data/SevenIdeaStrategy.json
```

### `EightBusinessModel.py`

Designs how the startup will create, deliver, and capture value. It defines the business model type, revenue streams, pricing logic, cost structure, key partners, key activities, resources, customer relationships, channels, and business model risks. Its output prepares the idea for unit economics and go-to-market planning.

Expected output file:

```text
data/EightBusinessModel.json
```

### `NineFunctionsList.py`

Translates the strategy into product capabilities. It defines what the product must do to deliver the value proposition, separates must-have functions from nice-to-have features, identifies future capabilities, and prevents feature creep before MVP planning. This agent should connect customer pain points, competitor gaps, and business model needs.

Expected output file:

```text
data/NineFunctionsList.json
```

### `TenMVPPlanning.py`

Plans the first buildable version of the product. It defines the MVP goal, riskiest assumptions, included and excluded scope, core user flows, milestones, testing plan, QA checklist, and launch criteria. The goal is to validate the startup idea with the smallest practical version instead of overbuilding.

Expected output file:

```text
data/TenMVPPlanning.json
```

### `ElevenUnitEconomicsAgent.py`

Checks whether the business can make financial sense. It estimates revenue model, pricing assumptions, cost assumptions, gross margin, CAC, LTV, LTV/CAC ratio, payback period, break-even point, and pricing tests. This agent is optional but important for understanding whether the idea can become economically viable.

Expected output file:

```text
data/ElevenUnitEconomicsAgent.json
```

### `TwelveGoToMarket.py`

Creates the launch and customer acquisition plan. It defines the target launch segment, positioning message, marketing channels, funnel stages, first experiments, success metrics, first 100 customers plan, launch timeline, CAC tracking, and feedback loops. Its output turns the MVP into a practical market entry plan.

Expected output file:

```text
data/TwelveGoToMarket.json
```

## Supporting Files

### `PipelineRunner.py`

Controls the pipeline execution. It should coordinate the agent sequence, pass each agent's output to the next step, save important results, and track pipeline progress. It is not a business-analysis agent by itself; it is the orchestrator for the agent workflow.

### `generalBot.py`

Represents a general-purpose assistant outside the main startup pipeline. It can be used for normal support, general questions, or non-pipeline conversations.

### `SkillsGap.py`

Can be used as a supporting analysis step to identify what the founder needs to learn or outsource. It is useful after the MVP, business model, or go-to-market plan reveals execution gaps.

## Section Refinement

Agents 4 through 12 should work as analysis agents with a small section-level chatbot. Each agent first generates its structured output, then the user can:

- regenerate the section with the same inputs;
- regenerate the section with a custom prompt;
- ask for a small change inside that section;
- keep the current result and continue to the next agent.

The refinement chat must stay inside the current section only. For example, an MVP refinement should update the MVP plan, not rewrite the customer analysis, competition analysis, or go-to-market plan.

## Data Folder

The `data` folder contains example output schemas and saved JSON structures for the pipeline. These files document the expected shape of each agent's result and make it easier to connect the backend, frontend, and database.

Important examples:

```text
data/questionnaireOutput.json
data/profileAnalysis.json
data/problems.json
data/IdeaIntake.json
data/FourCustomersAgent.json
data/FiveCompetitionAgent.json
data/SixMaketPotential.json
data/SevenIdeaStrategy.json
data/EightBusinessModel.json
data/NineFunctionsList.json
data/TenMVPPlanning.json
data/ElevenUnitEconomicsAgent.json
data/TwelveGoToMarket.json
```

## Database Integration

The backend schema should use the shared `agents` and `agent_runs` tables as the canonical storage for AI agent execution history. Each agent is registered once in `agents`, and every run is stored in `agent_runs` with its input JSON, output JSON, status, confidence score, execution time, and roadmap stage link when available.

Temporary per-agent result tables such as profile, problems, idea, or intake results may remain during transition for backward compatibility, but long-term agent outputs should be read from `agent_runs.output_data`.

## Product Logic

The project is designed to keep the founder moving through a structured startup-building journey:

1. Understand the founder or their existing idea.
2. Discover real problems and evidence.
3. Generate or refine a startup idea.
4. Save the selected idea.
5. Analyze customers and competition.
6. Estimate market potential.
7. Define strategy and business model.
8. Translate the idea into product functions.
9. Plan the MVP.
10. Check unit economics.
11. Build a go-to-market plan.

The final result should be more than an idea. It should be a practical startup plan with a clear customer, problem, value proposition, market logic, MVP scope, financial assumptions, and launch path.
