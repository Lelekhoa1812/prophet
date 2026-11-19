import pytest

from lab.source import staging


class Cur:
    """Scripted cursor: returns the next prepared row set for each execute()."""
    def __init__(self, *results):
        self.results, self.sql = list(results), []

    def execute(self, sql, *a):
        self.sql.append(sql)

    def fetchone(self):
        return self.results.pop(0)

    def fetchall(self):
        return self.results.pop(0)


def test_audit_flags_a_role_that_can_write():
    # flags row, membership rows, (writable relations, total relations)
    cur = Cur((False, True, True, True, True), [("neon_superuser",), ("axis_semantic_reader",)], (209, 209))
    a = staging.role_audit(cur)
    assert a["can_write"] is True and a["writable_relations"] == 209 and a["superuser_like"] is True and a["total_relations"] == 209


def test_audit_passes_a_reader_only_role():
    cur = Cur((False, False, False, False, False), [("axis_semantic_reader",)], (0, 209))
    a = staging.role_audit(cur)
    assert a["can_write"] is False and a["superuser_like"] is False


def test_every_lab_query_passes_the_statement_guard():
    for name, sql in staging.QUERIES.items():
        staging.check_select(sql)


@pytest.mark.parametrize("sql", ["delete from x", "select 1; drop table x", "  update t set a=1", "with a as (select 1) insert into t select * from a", "select 1 into newtable"])
def test_guard_refuses_writes_and_multiple_statements(sql):
    with pytest.raises(PermissionError):
        staging.check_select(sql)


def test_guard_is_not_an_assert():
    import inspect
    assert "assert " not in inspect.getsource(staging.check_select)         # asserts vanish under python -O
