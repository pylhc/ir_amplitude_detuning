import pytest

from ir_amplitude_detuning.detuning.terms import (
    FirstOrderTerm,
    SecondOrderTerm,
    detuning_term_to_planes,
    get_order,
)


@pytest.mark.parametrize(
    "term,expected",
    [
        (FirstOrderTerm.X10, 1),
        (FirstOrderTerm.X01, 1),
        (FirstOrderTerm.Y10, 1),
        (FirstOrderTerm.Y01, 1),
        (SecondOrderTerm.X20, 2),
        (SecondOrderTerm.X11, 2),
        (SecondOrderTerm.X02, 2),
        (SecondOrderTerm.Y20, 2),
        (SecondOrderTerm.Y11, 2),
        (SecondOrderTerm.Y02, 2),
        # also accept plain strings
        ("X11", 2),
        ("Y02", 2),
    ],
)
def test_get_order_various_terms(term, expected):
    assert get_order(term) == expected


@pytest.mark.parametrize(
    "term,expected_tune,expected_action",
    [
        (SecondOrderTerm.X02, "x", "yy"),
        (SecondOrderTerm.X11, "x", "xy"),
        (SecondOrderTerm.X20, "x", "xx"),
        (SecondOrderTerm.Y20, "y", "xx"),
        (FirstOrderTerm.Y01, "y", "y"),
        # string variants
        ("X10", "x", "x"),
        ("Y11", "y", "xy"),
    ],
)
def test_detuning_term_to_planes(term, expected_tune, expected_action):
    tune, action = detuning_term_to_planes(term)
    assert tune == expected_tune
    assert action == expected_action


@pytest.mark.parametrize(
    "bad_term,expected_exception",
    [
        ("X1", IndexError),       # too short -> indexing fails
        ("X1a", ValueError),      # non-digit in position -> int() fails
    ],
)
def test_get_order_raises_on_bad_input(bad_term, expected_exception):
    with pytest.raises(expected_exception):
        get_order(bad_term)

    with pytest.raises(expected_exception):
        detuning_term_to_planes(bad_term)
