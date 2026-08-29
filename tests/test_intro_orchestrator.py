from tasks.intro_orchestrator import _build_intro_steps


def test_build_intro_steps_follows_storyboard_order() -> None:
    content = {
        "processSteps": [
            {"index": 0, "title": "card0", "points": ["gesture", "voice"]},
            {"index": 1, "title": "card1", "points": ["five dimensions"]},
            {"index": 2, "title": "card2", "points": ["report"]},
        ],
        "dimensionConcepts": [
            {"id": "ssi", "index": 0, "explanation": "ssi line"},
            {"id": "reputation", "index": 1, "explanation": "rep line"},
        ],
        "deliverableConcepts": [
            {"id": "card", "index": 0, "concept": "card line"},
            {"id": "report", "index": 1, "concept": "report line"},
        ],
    }

    steps = _build_intro_steps(content)

    assert [s.target for s in steps] == [
        "intro_step",
        "intro_step",
        "intro_card_dimension",
        "intro_card_dimension",
        "intro_step",
        "intro_deliverable",
        "intro_deliverable",
    ]
    assert steps[0].index == 0 and steps[0].pace == "card"
    assert steps[1].index == 1 and steps[1].pace == "transition"
    assert steps[2].dimension_id == "ssi"
    assert steps[4].index == 2
    assert steps[5].dimension_id == "card"
    assert steps[0].anchors == ("card0", "gesture", "voice")
    assert steps[1].anchors == ("card1",)
    assert steps[1].fallback_speak == ""
