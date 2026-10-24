"""A scripted backend for tests. Not shipped and not a user-facing option.

The script maps an item's rendered state to what the backend should do with it:
either an exception to raise, or a dict of question id to (value, confidence).
For noul questions the confidence slot is ignored and value is the probability
of yes. Leaving a question out of the dict makes the answer go missing.
States are dict keys, so tests that use the stub need a string state_template.
"""

from jab_triage.backends.base import Answer, Backend, Decision, Usage


class ScriptedBackend(Backend):
    name = "stub"

    def __init__(self, script, model="stub-model", cost=0.001):
        self.script = script
        self.model = model
        self.cost = cost
        self.calls = []

    def decide(self, state, questions):
        self.calls.append(state)
        planned = self.script[state]
        if isinstance(planned, Exception):
            raise planned
        answers = {}
        for question in questions:
            if question.id in planned:
                answers[question.id] = make_answer(question, *planned[question.id])
        return Decision(
            answers=answers,
            usage=Usage(input_tokens=100, output_tokens=10, cost=self.cost),
            model=self.model,
        )


def make_answer(question, value, confidence=None):
    if question.type == "noul":
        return Answer("noul", value, {"true": value, "false": 1 - value}, None)
    return Answer(question.type, value, {str(value): confidence}, confidence)
