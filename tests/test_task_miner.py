"""Tests for task_miner module.

Covers: TaskRecord data model, JSONL I/O, diff splitting,
test name extraction, and instance ID generation.
No network calls — all GitHub interactions are tested via
subprocess mocking.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from task_miner import (
    TaskRecord,
    extract_test_names,
    split_diff,
    write_tasks,
)


# ── sample diffs ───────────────────────────────────────────────────

RUST_DIFF = textwrap.dedent("""\
    diff --git a/src/lib.rs b/src/lib.rs
    index aaa1111..bbb2222 100644
    --- a/src/lib.rs
    +++ b/src/lib.rs
    @@ -10,6 +10,7 @@ pub fn process(input: &str) -> Result<String, Error> {
         let trimmed = input.trim();
    +    if trimmed.is_empty() { return Err(Error::EmptyInput); }
         Ok(trimmed.to_uppercase())
     }
    diff --git a/tests/integration_test.rs b/tests/integration_test.rs
    index ccc3333..ddd4444 100644
    --- a/tests/integration_test.rs
    +++ b/tests/integration_test.rs
    @@ -5,6 +5,12 @@ use mylib::process;
    +#[test]
    +fn test_empty_input_returns_error() {
    +    let result = process("");
    +    assert!(result.is_err());
    +}
    +
    diff --git a/src/utils_test.rs b/src/utils_test.rs
    index eee5555..fff6666 100644
    --- a/src/utils_test.rs
    +++ b/src/utils_test.rs
    @@ -1,3 +1,9 @@
    +#[test]
    +fn test_trim_whitespace() {
    +    assert_eq!(trim("  hi  "), "hi");
    +}
""")

CSHARP_DIFF = textwrap.dedent("""\
    diff --git a/src/Services/OrderService.cs b/src/Services/OrderService.cs
    index aaa1111..bbb2222 100644
    --- a/src/Services/OrderService.cs
    +++ b/src/Services/OrderService.cs
    @@ -20,6 +20,8 @@ public class OrderService
         public Order CreateOrder(OrderRequest request)
         {
    +        if (request.Items.Count == 0)
    +            throw new ArgumentException("Order must have items");
             return new Order(request);
         }
    diff --git a/tests/OrderServiceTests.cs b/tests/OrderServiceTests.cs
    index ccc3333..ddd4444 100644
    --- a/tests/OrderServiceTests.cs
    +++ b/tests/OrderServiceTests.cs
    @@ -10,6 +10,20 @@ public class OrderServiceTests
    +    [Fact]
    +    public void CreateOrder_EmptyItems_ThrowsArgumentException()
    +    {
    +        var service = new OrderService();
    +        Assert.Throws<ArgumentException>(() => service.CreateOrder(new OrderRequest()));
    +    }
    +
    +    [Theory]
    +    [InlineData(0)]
    +    [InlineData(-1)]
    +    public void CreateOrder_InvalidQuantity_ThrowsArgumentException(int qty)
    +    {
    +        var service = new OrderService();
    +        Assert.Throws<ArgumentException>(() => service.CreateOrder(qty));
    +    }
""")

RUST_TOKIO_DIFF = textwrap.dedent("""\
    diff --git a/tests/async_test.rs b/tests/async_test.rs
    index aaa1111..bbb2222 100644
    --- a/tests/async_test.rs
    +++ b/tests/async_test.rs
    @@ -1,3 +1,9 @@
    +#[tokio::test]
    +async fn test_async_fetch() {
    +    let result = fetch("http://example.com").await;
    +    assert!(result.is_ok());
    +}
""")

CSHARP_NUNIT_DIFF = textwrap.dedent("""\
    diff --git a/Tests/CalculatorTest.cs b/Tests/CalculatorTest.cs
    index aaa1111..bbb2222 100644
    --- a/Tests/CalculatorTest.cs
    +++ b/Tests/CalculatorTest.cs
    @@ -1,3 +1,9 @@
    +    [Test]
    +    public void Add_TwoNumbers_ReturnsSum()
    +    {
    +        Assert.AreEqual(5, Calculator.Add(2, 3));
    +    }
    +
    +    [TestCase(1, 2, 3)]
    +    [TestCase(0, 0, 0)]
    +    public void Add_TestCases_ReturnsExpected(int a, int b, int expected)
    +    {
    +        Assert.AreEqual(expected, Calculator.Add(a, b));
    +    }
""")


# ── TaskRecord ─────────────────────────────────────────────────────


class TestTaskRecord:
    def test_frozen(self) -> None:
        record = TaskRecord(
            repo="owner/repo",
            base_commit="abc123",
            test_patch="diff ...",
            fail_to_pass=["test_one"],
            problem_statement="Bug description",
            lang="rust",
            instance_id="owner__repo-42",
            patch="diff ...",
        )
        with pytest.raises(AttributeError):
            record.repo = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        record = TaskRecord(
            repo="owner/repo",
            base_commit="abc123",
            test_patch="tp",
            fail_to_pass=["t1", "t2"],
            problem_statement="Fix it",
            lang="csharp",
            instance_id="owner__repo-99",
            patch="cp",
        )
        assert record.repo == "owner/repo"
        assert record.lang == "csharp"
        assert len(record.fail_to_pass) == 2


# ── JSONL Writer ───────────────────────────────────────────────────


class TestWriteTasks:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "tasks.jsonl"
        records = [
            TaskRecord(
                repo="o/r", base_commit="aaa", test_patch="tp",
                fail_to_pass=["t1"], problem_statement="fix",
                lang="rust", instance_id="o__r-1", patch="cp",
            ),
            TaskRecord(
                repo="o/r", base_commit="bbb", test_patch="tp2",
                fail_to_pass=["t2"], problem_statement="fix2",
                lang="rust", instance_id="o__r-2", patch="cp2",
            ),
        ]
        count = write_tasks(records, out)
        assert count == 2

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["repo"] == "o/r"
        assert first["fail_to_pass"] == ["t1"]

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        out = tmp_path / "tasks.jsonl"
        out.write_text('{"existing": true}\n')
        records = [
            TaskRecord(
                repo="o/r", base_commit="aaa", test_patch="tp",
                fail_to_pass=["t1"], problem_statement="fix",
                lang="rust", instance_id="o__r-1", patch="cp",
            ),
        ]
        write_tasks(records, out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_empty_records(self, tmp_path: Path) -> None:
        out = tmp_path / "tasks.jsonl"
        count = write_tasks([], out)
        assert count == 0
        assert not out.exists()


# ── Diff Splitter ──────────────────────────────────────────────────


class TestSplitDiffRust:
    def test_separates_test_and_code(self) -> None:
        test_patch, code_patch = split_diff(RUST_DIFF, "rust")
        assert "integration_test.rs" in test_patch
        assert "utils_test.rs" in test_patch
        assert "src/lib.rs" not in test_patch
        assert "src/lib.rs" in code_patch
        assert "integration_test.rs" not in code_patch

    def test_empty_diff(self) -> None:
        test_patch, code_patch = split_diff("", "rust")
        assert test_patch == ""
        assert code_patch == ""

    def test_no_test_files(self) -> None:
        diff = textwrap.dedent("""\
            diff --git a/src/lib.rs b/src/lib.rs
            --- a/src/lib.rs
            +++ b/src/lib.rs
            @@ -1,3 +1,4 @@
            +// comment
        """)
        test_patch, code_patch = split_diff(diff, "rust")
        assert test_patch == ""
        assert "lib.rs" in code_patch

    def test_only_test_files(self) -> None:
        diff = textwrap.dedent("""\
            diff --git a/tests/foo_test.rs b/tests/foo_test.rs
            --- a/tests/foo_test.rs
            +++ b/tests/foo_test.rs
            @@ -1,3 +1,4 @@
            +#[test]
        """)
        test_patch, code_patch = split_diff(diff, "rust")
        assert "foo_test.rs" in test_patch
        assert code_patch == ""


class TestSplitDiffCSharp:
    def test_separates_test_and_code(self) -> None:
        test_patch, code_patch = split_diff(CSHARP_DIFF, "csharp")
        assert "OrderServiceTests.cs" in test_patch
        assert "OrderService.cs" not in test_patch
        assert "OrderService.cs" in code_patch

    def test_nunit_test_file(self) -> None:
        test_patch, code_patch = split_diff(CSHARP_NUNIT_DIFF, "csharp")
        assert "CalculatorTest.cs" in test_patch
        assert code_patch == ""


# ── Test Name Extraction ──────────────────────────────────────────


class TestExtractTestNamesRust:
    def test_standard_tests(self) -> None:
        _, test_patch = split_diff(RUST_DIFF, "rust")
        # Use the full diff's test portion
        test_patch_full, _ = split_diff(RUST_DIFF, "rust")
        names = extract_test_names(test_patch_full, "rust")
        assert "test_empty_input_returns_error" in names
        assert "test_trim_whitespace" in names

    def test_tokio_test(self) -> None:
        names = extract_test_names(RUST_TOKIO_DIFF, "rust")
        assert "test_async_fetch" in names

    def test_empty_patch(self) -> None:
        assert extract_test_names("", "rust") == []

    def test_no_added_tests(self) -> None:
        patch = textwrap.dedent("""\
            diff --git a/tests/test.rs b/tests/test.rs
            --- a/tests/test.rs
            +++ b/tests/test.rs
            @@ -1,3 +1,4 @@
            +// just a comment
        """)
        assert extract_test_names(patch, "rust") == []


class TestExtractTestNamesCSharp:
    def test_xunit_fact_and_theory(self) -> None:
        test_patch, _ = split_diff(CSHARP_DIFF, "csharp")
        names = extract_test_names(test_patch, "csharp")
        assert "CreateOrder_EmptyItems_ThrowsArgumentException" in names
        assert "CreateOrder_InvalidQuantity_ThrowsArgumentException" in names

    def test_nunit_test_and_testcase(self) -> None:
        names = extract_test_names(CSHARP_NUNIT_DIFF, "csharp")
        assert "Add_TwoNumbers_ReturnsSum" in names
        assert "Add_TestCases_ReturnsExpected" in names

    def test_empty_patch(self) -> None:
        assert extract_test_names("", "csharp") == []


# ── Instance ID ────────────────────────────────────────────────────


class TestInstanceId:
    def test_format(self) -> None:
        from task_miner import make_instance_id

        assert make_instance_id("owner/repo", 42) == "owner__repo-42"

    def test_org_with_dash(self) -> None:
        from task_miner import make_instance_id

        assert make_instance_id("my-org/my-repo", 7) == "my-org__my-repo-7"
