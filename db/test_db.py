from db.connection import Base, engine, get_session, test_connection
from db import crud, models


def main() -> int:
    if not test_connection():
        return 1

    Base.metadata.create_all(bind=engine)
    print("Tables ensured:", ", ".join(sorted(Base.metadata.tables.keys())))

    test_user_id = "db_test_connection_user"
    test_questionnaire = {
        "user_profile": {
            "curiosity_domain": "Business",
            "experience_level": "Steady",
            "business_interests": ["Marketplace"],
            "target_region": "MENA",
            "founder_setup": "Solo",
            "risk_tolerance": "Moderate",
        },
        "career_profile": {
            "preferred_work_types": ["Working with ideas"],
            "desired_impact": ["Make strong income"],
        },
    }

    with get_session() as session:
        crud.upsert_pipeline_status(session, test_user_id, "pending", "connection_test")
        row = crud.get_pipeline_status(session, test_user_id)

        if row is None:
            print("CRUD test failed: test row was not found")
            return 1

        print("CRUD insert/read OK:", row.user_id, row.status, row.current_step)

        crud.save_questionnaire_output(session, test_user_id, test_questionnaire)
        saved_questionnaire = crud.get_questionnaire_output_json(session, test_user_id)
        if saved_questionnaire != test_questionnaire:
            print("Questionnaire test failed: saved JSON did not match")
            return 1

        print("Questionnaire JSON save/read OK")

        questionnaire_row = crud.get_questionnaire_output(session, test_user_id)
        if questionnaire_row is not None:
            session.delete(questionnaire_row)
        session.delete(row)
        print("Cleanup OK: deleted test row")

    print("Database integration test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
