import pytest

from chronify.exceptions import InvalidParameter
from chronify.time import (
    TimeZone,
    get_standard_time,
    get_prevailing_time,
    get_time_zone_offset,
    is_prevailing,
    is_standard,
)


def test_standard_time():
    assert get_standard_time(TimeZone.UTC) == TimeZone.UTC
    assert get_standard_time(TimeZone.HST) == TimeZone.HST
    assert get_standard_time(TimeZone.AST) == TimeZone.AST
    assert get_standard_time(TimeZone.APT) == TimeZone.AST
    assert get_standard_time(TimeZone.PST) == TimeZone.PST
    assert get_standard_time(TimeZone.PPT) == TimeZone.PST
    assert get_standard_time(TimeZone.MST) == TimeZone.MST
    assert get_standard_time(TimeZone.MPT) == TimeZone.MST
    assert get_standard_time(TimeZone.CST) == TimeZone.CST
    assert get_standard_time(TimeZone.CPT) == TimeZone.CST
    assert get_standard_time(TimeZone.EST) == TimeZone.EST
    assert get_standard_time(TimeZone.EPT) == TimeZone.EST
    assert get_standard_time(TimeZone.ARIZONA) == TimeZone.ARIZONA


def test_prevailing_time():
    assert get_prevailing_time(TimeZone.UTC) == TimeZone.UTC
    assert get_prevailing_time(TimeZone.HST) == TimeZone.HST
    assert get_prevailing_time(TimeZone.AST) == TimeZone.APT
    assert get_prevailing_time(TimeZone.APT) == TimeZone.APT
    assert get_prevailing_time(TimeZone.PST) == TimeZone.PPT
    assert get_prevailing_time(TimeZone.PPT) == TimeZone.PPT
    assert get_prevailing_time(TimeZone.MST) == TimeZone.MPT
    assert get_prevailing_time(TimeZone.MPT) == TimeZone.MPT
    assert get_prevailing_time(TimeZone.CST) == TimeZone.CPT
    assert get_prevailing_time(TimeZone.CPT) == TimeZone.CPT
    assert get_prevailing_time(TimeZone.EST) == TimeZone.EPT
    assert get_prevailing_time(TimeZone.EPT) == TimeZone.EPT
    assert get_prevailing_time(TimeZone.ARIZONA) == TimeZone.ARIZONA


def test_is_standard():
    assert is_standard(TimeZone.EST)
    assert not is_standard(TimeZone.EPT)
    assert is_prevailing(TimeZone.EPT)
    assert not is_prevailing(TimeZone.EST)


def test_get_time_zone_offset():
    assert get_time_zone_offset(TimeZone.EST) == "-05:00"
    assert get_time_zone_offset(TimeZone.MST) == "-07:00"
    with pytest.raises(InvalidParameter):
        assert get_time_zone_offset(TimeZone.EPT)
