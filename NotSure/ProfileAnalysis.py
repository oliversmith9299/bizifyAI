###1 profile analysis (Analysis not chatbot)
#Assess founder suitability
#Build personal runway 
#chat to founder to get more ideas if those are not what they want
######

#input is questionnaireoutput.json and then will output a json insights to be an input for the next agent (problem discovery)
#  and also will be stored in the database to be used by the future agents in the flow.

#rule-based

def profile_analysis(data):
    user = data["user_profile"]
    career = data["career_profile"]

    industries = user["business_interests"]

    # Domains
    domains = []
    if user["curiosity_domain"] == "Technology":
        domains = ["digital platforms", "technology tools"]

    # Customers
    customers = []
    if "E-commerce" in industries:
        customers.append("small online sellers")
    if "Marketplace" in industries:
        customers.append("platform buyers and sellers")

    # Work style
    work_style = []
    if "Working independently" in career["preferred_work_types"]:
        work_style.append("independent")
    if "Working with technology" in career["preferred_work_types"]:
        work_style.append("technology-oriented")

    # Risk mapping
    risk_map = {
        "I prefer stability and predictable outcomes": "low",
        "I'm comfortable with moderate risk if there's growth potential": "moderate",
        "I enjoy taking calculated risks": "moderate-high",
        "I thrive in high-risk, high-reward situations": "high"
    }

    risk = risk_map.get(user["risk_tolerance"], "moderate")

    # Suitability score
    score = 0
    if risk in ["moderate", "moderate-high", "high"]:
        score += 1
    if user["experience_level"].lower() != "beginner":
        score += 1
    if "independent" in work_style:
        score += 1

    suitability = "high" if score >= 2 else "medium" if score == 1 else "low"

    # Runway estimate
    if risk == "low":
        runway = "needs stable income"
    elif risk == "moderate":
        runway = "can handle part-time startup"
    else:
        runway = "can take full startup risk"

    # Keywords
    keywords = []
    for industry in industries:
        keywords.append(f"{industry.lower()} problems {user['target_region']}")
        keywords.append(f"{industry.lower()} challenges {user['target_region']}")

    # Flags
    needs_guidance = user["experience_level"].lower() == "beginner"

    return {
        "industries": industries,
        "domains": domains,
        "target_region": user["target_region"],
        "customer_focus": customers,
        "founder_profile": {
            "experience_level": user["experience_level"].lower(),
            "risk_level": risk,
            "work_style": work_style,
            "suitability_score": suitability,
            "runway_estimate": runway
        },
        "business_preferences": {
            "model_preferences": industries,
            "team_preference": user["founder_setup"].lower()
        },
        "search_context": {
            "keywords": keywords
        },
        "system_flags": {
            "needs_guidance": needs_guidance,
            "should_trigger_chat": needs_guidance
        }
    }

