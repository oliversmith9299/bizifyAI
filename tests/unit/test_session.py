"""
Unit tests for orchestrator/session.py
========================================
Tests the PipelineSession state machine — no DB or LLM involved.
"""

import pytest

from orchestrator.session import (
    Flow,
    PipelineSession,
    Step,
    make_session,
)


# ── Construction ──────────────────────────────────────────────────────────────

class TestPipelineSessionConstruction:
    def test_new_user_starts_at_profile_analysis(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        assert session.current_step == Step.PROFILE_ANALYSIS

    def test_returning_user_starts_at_idea_intake(self):
        session = PipelineSession(user_id="u1", flow=Flow.RETURNING_USER)
        assert session.current_step == Step.IDEA_INTAKE

    def test_results_start_empty(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        assert session.results == {}

    def test_error_starts_none(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        assert session.error is None


# ── advance() ─────────────────────────────────────────────────────────────────

class TestAdvance:
    def test_advance_moves_to_next_step(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        assert session.current_step == Step.PROFILE_ANALYSIS
        session.advance()
        assert session.current_step == Step.PROBLEM_DISCOVERY

    def test_advance_stores_result(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        session.advance(result={"data": "profile_output"})
        assert session.results[Step.PROFILE_ANALYSIS.value] == {"data": "profile_output"}

    def test_advance_with_none_result_does_not_store(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        session.advance(result=None)
        assert Step.PROFILE_ANALYSIS.value not in session.results

    def test_advance_through_full_new_user_flow_reaches_done(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        while not session.is_done():
            session.advance()
        assert session.current_step == Step.DONE

    def test_advance_through_full_returning_user_flow_reaches_done(self):
        session = PipelineSession(user_id="u1", flow=Flow.RETURNING_USER)
        while not session.is_done():
            session.advance()
        assert session.current_step == Step.DONE

    def test_advance_at_done_step_stays_done(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        while not session.is_done():
            session.advance()
        session.advance()
        assert session.current_step == Step.DONE


# ── go_to() ───────────────────────────────────────────────────────────────────

class TestGoTo:
    def test_go_to_valid_step(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        session.go_to(Step.CUSTOMERS)
        assert session.current_step == Step.CUSTOMERS

    def test_go_to_invalid_step_raises_value_error(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        with pytest.raises(ValueError, match="not valid"):
            session.go_to(Step.IDEA_INTAKE)

    def test_go_to_returns_self_for_chaining(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        result = session.go_to(Step.COMPETITION)
        assert result is session


# ── fail() ────────────────────────────────────────────────────────────────────

class TestFail:
    def test_fail_sets_error_message(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        session.fail("Something went wrong")
        assert session.error == "Something went wrong"

    def test_fail_does_not_advance_step(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        original_step = session.current_step
        session.fail("error")
        assert session.current_step == original_step

    def test_fail_returns_self_for_chaining(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        result = session.fail("error")
        assert result is session


# ── is_done / is_failed ───────────────────────────────────────────────────────

class TestStatusQueries:
    def test_is_done_initially_false(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        assert not session.is_done()

    def test_is_done_true_when_at_done_step(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        session.go_to(Step.DONE)
        assert session.is_done()

    def test_is_failed_initially_false(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        assert not session.is_failed()

    def test_is_failed_true_after_fail(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        session.fail("oops")
        assert session.is_failed()


# ── Properties ────────────────────────────────────────────────────────────────

class TestProperties:
    def test_step_label_is_human_readable(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        assert session.step_label == "Profile Analysis"

    def test_step_label_changes_with_step(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        session.go_to(Step.CUSTOMERS)
        assert session.step_label == "Customer Analysis"

    def test_agent_returns_dotted_path(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        assert "PipelineRunner" in session.agent

    def test_remaining_steps_starts_with_current(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        session.go_to(Step.COMPETITION)
        remaining = session.remaining_steps()
        assert remaining[0] == Step.COMPETITION

    def test_remaining_steps_ends_with_done(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        remaining = session.remaining_steps()
        assert remaining[-1] == Step.DONE


# ── Serialisation ─────────────────────────────────────────────────────────────

class TestSerialisation:
    def test_to_dict_contains_required_keys(self):
        session = PipelineSession(user_id="u1", flow=Flow.NEW_USER)
        d = session.to_dict()
        assert "user_id" in d
        assert "flow" in d
        assert "current_step" in d

    def test_from_dict_restores_session(self):
        original = PipelineSession(user_id="u42", flow=Flow.NEW_USER)
        original.go_to(Step.CUSTOMERS)
        original.fail("test error")

        restored = PipelineSession.from_dict(original.to_dict())
        assert restored.user_id == "u42"
        assert restored.current_step == Step.CUSTOMERS
        assert restored.error == "test error"

    def test_from_dict_round_trip_preserves_flow(self):
        session = PipelineSession(user_id="u1", flow=Flow.RETURNING_USER)
        restored = PipelineSession.from_dict(session.to_dict())
        assert restored.flow == Flow.RETURNING_USER


# ── make_session factory ──────────────────────────────────────────────────────

class TestMakeSession:
    def test_with_idea_creates_returning_user_session(self):
        session = make_session(user_id="u1", has_idea=True)
        assert session.flow == Flow.RETURNING_USER
        assert session.current_step == Step.IDEA_INTAKE

    def test_without_idea_creates_new_user_session(self):
        session = make_session(user_id="u1", has_idea=False)
        assert session.flow == Flow.NEW_USER
        assert session.current_step == Step.PROFILE_ANALYSIS
