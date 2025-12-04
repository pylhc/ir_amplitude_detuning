from __future__ import annotations

from unittest.mock import Mock

import pytest

from ir_amplitude_detuning.utilities.common import (
    BeamDict,
    Container,
    ContainerMeta,
    StrEnum,
    to_loop,
)

# ============================================================================
# Tests for StrEnum
# ============================================================================

class TestStrEnum:
    """Test cases for the StrEnum class."""

    def test_str_enum_creation(self):
        """Test creating a StrEnum with string values."""
        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"

        assert Color.RED.value == "red"
        assert Color.BLUE.value == "blue"

    def test_str_enum_compare_with_enum(self):
        """Test StrEnum members can be compared."""
        class Color(StrEnum):
            RED = "red"
            GREEN = "green"

        assert Color.RED == Color.RED
        assert Color.RED != Color.GREEN
        assert Color("red") == Color.RED

    def test_str_enum_compare_with_string(self):
        """Test creating a StrEnum with string values."""
        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"

        assert Color.RED == "red"
        assert Color.BLUE == "blue"
        assert Color.BLUE != "red"

    def test_str_enum_is_iterable(self):
        """Test creating a StrEnum with string values."""
        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"

        # assert "red" in Color  # py3.12+
        assert "red" in list(Color)

    def test_str_enum_not_member(self):
        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"

        assert "green" not in list(Color)
        with pytest.raises(ValueError):
            Color("green")

    def test_str_enum_str(self):
        """Test __str__ returns the string value."""
        class Status(StrEnum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        assert str(Status.ACTIVE) == "active"
        assert str(Status.INACTIVE) == "inactive"

    def test_str_enum_string_conversion_in_f_string(self):
        """Test StrEnum works in f-strings."""
        class Status(StrEnum):
            ACTIVE = "active"

        result = f"Status: {Status.ACTIVE}"
        assert result == "Status: active"


# ============================================================================
# Tests for ContainerMeta
# ============================================================================

class TestContainerMeta:
    """Test cases for the ContainerMeta metaclass."""

    def test_container_meta_getitem(self):
        """Test __getitem__ retrieves class attributes."""
        class MyContainer(metaclass=ContainerMeta):
            attr1 = "value1"
            attr2 = "value2"

        assert MyContainer["attr1"] == "value1"
        assert MyContainer["attr2"] == "value2"

    def test_container_meta_getitem_missing(self):
        """Test __getitem__ raises KeyError for missing attribute."""
        class MyContainer(metaclass=ContainerMeta):
            attr1 = "value1"

        with pytest.raises(KeyError):
            MyContainer["nonexistent"]

    def test_container_meta_iter(self):
        """Test __iter__ iterates over public attributes only."""
        class MyContainer(metaclass=ContainerMeta):
            public1 = "value1"
            public2 = "value2"
            _private = "hidden"

        keys = list(MyContainer)
        assert "public1" in keys
        assert "public2" in keys
        assert "_private" not in keys

        assert all(k1 == k2 for k1, k2 in zip(keys, MyContainer.keys()))

    def test_container_meta_len(self):
        """Test __len__ returns count of public attributes."""
        class MyContainer(metaclass=ContainerMeta):
            attr1 = "value1"
            attr2 = "value2"
            attr3 = "value3"
            _private = "hidden"

        assert len(MyContainer) == 3

    def test_container_meta_keys(self):
        """Test keys() method returns iterable of public keys."""
        class MyContainer(metaclass=ContainerMeta):
            key1 = "value1"
            key2 = "value2"
            _private = "hidden"

        keys = MyContainer.keys()
        assert "key1" in keys
        assert "key2" in keys
        assert "_private" not in keys
        assert len(keys) == 2

        assert all(k1 == k2 for k1, k2 in zip(keys, MyContainer))

    def test_container_meta_excludes_metaclass_methods(self):
        """Test that metaclass methods are not included in iteration."""
        class MyContainer(metaclass=ContainerMeta):
            attr = "value"

        keys = list(MyContainer)
        assert "__getitem__" not in keys
        assert "__iter__" not in keys
        assert "__len__" not in keys
        assert "keys" not in keys


# ============================================================================
# Tests for Container
# ============================================================================

class TestContainer:
    """Test cases for the Container class."""

    def test_container_inheritable(self):
        """Test Container can be inherited."""
        class MyContainer(Container):
            attr1 = "value1"
            attr2 = "value2"

        assert MyContainer["attr1"] == "value1"
        assert MyContainer["attr2"] == "value2"
        assert len(MyContainer.keys()) == 2

    def test_container_has_metaclass(self):
        """Test Container uses ContainerMeta metaclass."""
        assert type(Container) is ContainerMeta

    def test_container_iteration(self):
        """Test iteration works on inherited Container."""
        class MyContainer(Container):
            item1 = "first"
            item2 = "second"
            _item3 = "third"

        # via list
        items = list(MyContainer)
        assert len(items) == 2
        assert "item1" in items
        assert "item2" in items
        assert "_item3" not in items

        # or direct
        assert len(MyContainer) == 2
        assert "item1" in MyContainer
        assert "item2" in MyContainer
        assert "_item3" not in MyContainer


# ============================================================================
# Tests for BeamDict
# ============================================================================

class TestBeamDict:
    """Test cases for the BeamDict class."""

    def test_beam_dict_basic_access(self):
        """Test basic dictionary access."""
        bd = BeamDict({1: "beam1", 2: "beam2"})
        assert bd[1] == "beam1"
        assert bd[2] == "beam2"

    def test_beam_dict_beam_2_4_interchangeable_access_2(self):
        """Test beam 2 accesses beam 4 data when 2 not present."""
        bd = BeamDict({1: "beam1", 4: "beam4"})
        assert bd[2] == "beam4"

    def test_beam_dict_beam_2_4_interchangeable_access_4(self):
        """Test beam 4 accesses beam 2 data when 4 not present."""
        bd = BeamDict({1: "beam1", 2: "beam2"})
        assert bd[4] == "beam2"

    def test_beam_dict_beam_2_4_preference(self):
        """Test that present beam takes precedence."""
        bd = BeamDict({1: "beam1", 2: "beam2", 4: "beam4"})
        assert bd[2] == "beam2"
        assert bd[4] == "beam4"

    def test_beam_dict_missing_key_no_default(self):
        """Test KeyError raised for missing beam with no default."""
        bd = BeamDict({1: "beam1"})
        with pytest.raises(KeyError, match="Beam 3 not defined"):
            bd[3]

    def test_beam_dict_missing_key_with_default(self):
        """Test default factory called for missing key."""
        default_factory = Mock(return_value="default_value")
        bd = BeamDict.from_dict({1: "beam1"}, default=default_factory)

        result = bd[3]

        assert result == "default_value"
        default_factory.assert_called_once()

    def test_beam_dict_from_dict(self):
        """Test from_dict classmethod creates BeamDict."""
        bd = BeamDict.from_dict({1: 10, 2: 20})
        assert bd[1] == 10
        assert bd[2] == 20

    def test_beam_dict_add(self):
        """Test __add__ operation."""
        bd1 = BeamDict({1: 10, 2: 20})
        bd2 = BeamDict({1: 5, 2: 15})

        result = bd1 + bd2

        assert isinstance(result, BeamDict)
        assert result[1] == 15
        assert result[2] == 35

    def test_beam_dict_sub(self):
        """Test __sub__ operation."""
        bd1 = BeamDict({1: 10, 2: 20})
        bd2 = BeamDict({1: 3, 2: 5})

        result = bd1 - bd2

        assert isinstance(result, BeamDict)
        assert result[1] == 7
        assert result[2] == 15

    def test_beam_dict_mul(self):
        """Test __mul__ operation."""
        bd = BeamDict({1: 10, 2: 20})

        result = bd * 2

        assert isinstance(result, BeamDict)
        assert result[1] == 20
        assert result[2] == 40

    def test_beam_dict_rmul(self):
        """Test __rmul__ right multiplication operation."""
        bd = BeamDict({1: 10, 2: 20})

        result = 3 * bd

        assert isinstance(result, BeamDict)
        assert result[1] == 30
        assert result[2] == 60

    def test_beam_dict_truediv(self):
        """Test __truediv__ division operation."""
        bd = BeamDict({1: 10.0, 2: 20.0})

        result = bd / 2

        assert isinstance(result, BeamDict)
        assert result[1] == pytest.approx(5.0)
        assert result[2] == pytest.approx(10.0)

    def test_beam_dict_arithmetic_with_floats(self):
        """Test arithmetic operations with float values."""
        bd1 = BeamDict({1: 1.5, 2: 2.5})
        bd2 = BeamDict({1: 0.5, 2: 0.5})

        add_result = bd1 + bd2
        assert add_result[1] == pytest.approx(2.0)
        assert add_result[2] == pytest.approx(3.0)

        sub_result = bd1 - bd2
        assert sub_result[1] == pytest.approx(1.0)
        assert sub_result[2] == pytest.approx(2.0)

        mul_result = bd1 * 2
        assert mul_result[1] == pytest.approx(3.0)
        assert mul_result[2] == pytest.approx(5.0)

        div_result = bd1 / 2
        assert div_result[1] == pytest.approx(0.75)
        assert div_result[2] == pytest.approx(1.25)

    def test_beam_dict_arithmetic_returns_new_instance(self):
        """Test that arithmetic operations return new BeamDict instances."""
        bd1 = BeamDict({1: 10, 2: 20})
        bd2 = BeamDict({1: 5, 2: 15})

        result = bd1 + bd2

        assert result is not bd1
        assert result is not bd2
        assert bd1[1] == 10  # Original unchanged


# ============================================================================
# Tests for to_loop function
# ============================================================================

class TestToLoop:
    """Test cases for the to_loop function."""

    def test_to_loop_single_element(self):
        """Test to_loop with single-element iterable."""
        result = to_loop([1])
        assert len(result) == 1
        assert result[0] == [1]

    def test_to_loop_multiple_elements(self):
        """Test to_loop with multiple-element iterable."""
        result = to_loop([1, 2, 3])
        assert len(result) == 4
        assert result[0] == [1, 2, 3]
        assert result[1] == [1]
        assert result[2] == [2]
        assert result[3] == [3]

    def test_to_loop_two_elements(self):
        """Test to_loop with two-element iterable."""
        result = to_loop(["a", "b"])
        assert len(result) == 3
        assert result[0] == ["a", "b"]
        assert result[1] == ["a"]
        assert result[2] == ["b"]

    def test_to_loop_with_tuple(self):
        """Test to_loop works with tuples."""
        result = to_loop((1, 2))
        assert len(result) == 3
        assert result[0] == (1, 2)
        assert result[1] == [1]
        assert result[2] == [2]

    def test_to_loop_with_empty_list_raises(self):
        """Test to_loop with empty list."""
        with pytest.raises(ValueError, match="[Nn]othing"):
            to_loop([])
