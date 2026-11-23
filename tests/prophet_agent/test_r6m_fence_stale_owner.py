"""Round 6m (2026-11-23): an outside write (the service's cancel, a budget change) is refused only while another runtime holds
a live lease. Before the fix, a lease that another runtime had taken and then let lapse still refused the outside write,
because the database keeps the old owner's name until a sweep clears it. All missions here are synthetic.
"""
import pytest

from prophet_agent.runtime.store import LeaseLost, Store


@pytest.fixture()
def path(tmp_path):
    return tmp_path / "prophet.sqlite"


def _lapse(store, mid):
    store.conn.execute("UPDATE missions SET lease_until=0 WHERE id=?", (mid,))


def test_an_outside_write_passes_once_the_runtime_that_took_over_has_lapsed(path):
    a, b = Store(path), Store(path)
    try:
        mid = a.create_mission("forecast", {})
        a.acquire_lease(mid, "a", ttl_s=3600)
        _lapse(a, mid)                                    # a stalls past its lease
        b.acquire_lease(mid, "b", ttl_s=3600)             # b takes the mission over
        _lapse(b, mid)                                    # b's lease lapses too; nobody holds it now
        a.set_status(mid, "cancelled")                    # an outside write through a's store must not be refused
        assert a.mission(mid)["status"] == "cancelled"
    finally:
        a.close()
        b.close()


def test_an_outside_write_is_still_refused_while_another_runtime_holds_a_live_lease(path):
    a, b = Store(path), Store(path)
    try:
        mid = a.create_mission("forecast", {})
        a.acquire_lease(mid, "a", ttl_s=3600)
        _lapse(a, mid)
        b.acquire_lease(mid, "b", ttl_s=3600)             # b holds a live lease
        with pytest.raises(LeaseLost):
            a.set_status(mid, "cancelled")
        assert b.mission(mid)["status"] != "cancelled"
    finally:
        a.close()
        b.close()
