"""Panel SPX (2026-11-26): ranges of one total that differ by under 1 percent. A figure belongs to the range it is nearest."""
from prophet_agent.runtime.grounding import bound_misnamed

SENTENCE = "The 80% range for the 20-day total is 132,064 to 145,108 (average per step 6,929, range 6,603 to 7,255)."
TOGETHER = (130_785, 146_388)
MEASURED = (132_064, 145_108)


def test_the_measured_range_is_not_read_as_the_nearby_together_bound():
    assert bound_misnamed(SENTENCE, *TOGETHER, True, (MEASURED,)) is None


def test_the_together_bound_itself_is_still_caught():
    text = "The 80% range for the 20-day total is 130,785 to 146,388."
    assert bound_misnamed(text, *TOGETHER, True, (MEASURED,)) == text


def test_without_other_ranges_the_old_one_percent_rule_holds():
    assert bound_misnamed(SENTENCE, *TOGETHER, True) == SENTENCE
