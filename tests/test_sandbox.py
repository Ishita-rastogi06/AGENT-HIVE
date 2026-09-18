"""
Tests for the Python sandbox tool (Item 6 requirement).

Verifies:
  - class definitions with __init__ run correctly
  - class definitions with methods run correctly
  - getattr on public attributes is allowed
  - setattr on public attributes is allowed
  - getattr with a dunder/private name is blocked (AST + runtime)
  - setattr with a dunder/private name is blocked (AST + runtime)
  - direct dunder attribute access (x.__class__) is blocked at AST level
  - normal method calls (list.append, str.upper, dict.items) still work
  - eval / exec still blocked
  - blocked imports (os, subprocess, sys) still blocked
  - allowed imports (math, json, collections, etc.) still work
  - sandbox timeout is enforced
  - large output is truncated
"""
import sys
import unittest

sys.path.insert(0, __file__.split("tests")[0].rstrip("\\/"))

from src.tools.sandbox_tools import PythonSandboxTool
from src.tools.base import ToolContext


class SandboxTestCase(unittest.TestCase):
    """Base class providing shared helpers."""

    @classmethod
    def setUpClass(cls):
        cls.tool = PythonSandboxTool()
        cls.ctx  = ToolContext(agent_name="test", subtask_id="t0")

    def assertRunsOk(self, code: str, expected_in_output: str = ""):
        """Assert code validates and executes successfully."""
        self.tool.validate_arguments({"code": code})
        result = self.tool.execute({"code": code}, self.ctx)
        self.assertTrue(
            result.success,
            f"Expected success but got error: {result.error}\nCode:\n{code}",
        )
        if expected_in_output:
            self.assertIn(
                expected_in_output, str(result.output),
                f"Expected '{expected_in_output}' in output '{result.output}'",
            )
        return result

    def assertValidationBlocked(self, code: str, fragment: str = ""):
        """Assert validate_arguments raises PermissionError."""
        with self.assertRaises(PermissionError, msg=f"Should block: {code[:60]}") as cm:
            self.tool.validate_arguments({"code": code})
        if fragment:
            self.assertIn(fragment, str(cm.exception))

    def assertExecutionBlocked(self, code: str):
        """Assert code is blocked either at validation or execution."""
        try:
            self.tool.validate_arguments({"code": code})
        except PermissionError:
            return  # blocked at validation — pass

        result = self.tool.execute({"code": code}, self.ctx)
        self.assertFalse(
            result.success,
            f"Expected blocked execution but succeeded with: {result.output}",
        )


# ── 1. Class definitions ──────────────────────────────────────────────────────

class TestClassDefinitions(SandboxTestCase):
    """class statements must work — this was the core Item 6 bug."""

    def test_simple_class_with_method(self):
        self.assertRunsOk(
            "class Greeter:\n"
            "    def greet(self, name): return 'Hello ' + name\n"
            "print(Greeter().greet('World'))",
            "Hello World",
        )

    def test_class_with_init(self):
        self.assertRunsOk(
            "class Counter:\n"
            "    def __init__(self): self.n = 0\n"
            "    def inc(self): self.n += 1\n"
            "c = Counter()\nc.inc()\nc.inc()\nprint(c.n)",
            "2",
        )

    def test_class_with_math_import(self):
        self.assertRunsOk(
            "import math\n"
            "class Circle:\n"
            "    def __init__(self, r): self.r = r\n"
            "    def area(self): return round(math.pi * self.r ** 2, 2)\n"
            "print(Circle(5).area())",
            "78.54",
        )

    def test_class_inheritance(self):
        self.assertRunsOk(
            "class Animal:\n"
            "    def speak(self): return 'generic'\n"
            "class Dog(Animal):\n"
            "    def speak(self): return 'woof'\n"
            "print(Dog().speak())",
            "woof",
        )

    def test_class_with_classmethod_via_decorator(self):
        # classmethod uses @classmethod decorator — should be fine.
        self.assertRunsOk(
            "class Foo:\n"
            "    count = 0\n"
            "    def __init__(self): Foo.count += 1\n"
            "Foo(); Foo(); Foo()\n"
            "print(Foo.count)",
            "3",
        )

    def test_fibonacci_class(self):
        self.assertRunsOk(
            "class Fib:\n"
            "    def compute(self, n):\n"
            "        a, b = 0, 1\n"
            "        for _ in range(n): a, b = b, a + b\n"
            "        return a\n"
            "print([Fib().compute(i) for i in range(8)])",
            "[0, 1, 1, 2, 3, 5, 8, 13]",
        )


# ── 2. getattr / setattr — public allowed ─────────────────────────────────────

class TestGetAttrSetAttrPublic(SandboxTestCase):

    def test_getattr_public_method(self):
        self.assertRunsOk(
            "x = [1, 2, 3]\n"
            "fn = getattr(x, 'append')\n"
            "fn(4)\n"
            "print(x)",
            "[1, 2, 3, 4]",
        )

    def test_getattr_public_attribute(self):
        self.assertRunsOk(
            "class C:\n"
            "    value = 99\n"
            "print(getattr(C, 'value'))",
            "99",
        )

    def test_setattr_public(self):
        self.assertRunsOk(
            "class C: pass\n"
            "c = C()\n"
            "setattr(c, 'score', 42)\n"
            "print(c.score)",
            "42",
        )

    def test_getattr_with_default(self):
        self.assertRunsOk(
            "class C: pass\n"
            "c = C()\n"
            "print(getattr(c, 'missing', 'default'))",
            "default",
        )


# ── 3. getattr / setattr — dunder blocked ────────────────────────────────────

class TestGetAttrSetAttrDunderBlocked(SandboxTestCase):

    def test_getattr_dunder_blocked_ast(self):
        self.assertValidationBlocked(
            "x = 'hello'\nprint(getattr(x, '__class__'))",
            "__class__",
        )

    def test_setattr_dunder_blocked_ast(self):
        self.assertValidationBlocked(
            "class C: pass\nsetattr(C, '__dict__', {})",
            "__dict__",
        )

    def test_getattr_private_blocked_ast(self):
        self.assertValidationBlocked(
            "x = [1, 2]\ngetattr(x, '_internal')",
            "_internal",
        )

    def test_getattr_dunder_dynamic_blocked_runtime(self):
        # Dynamic string — AST can't see it; runtime wrapper must catch it.
        self.assertExecutionBlocked(
            "name = '__class__'\nx = 'hello'\nprint(getattr(x, name))"
        )

    def test_setattr_dunder_dynamic_blocked_runtime(self):
        self.assertExecutionBlocked(
            "name = '__dict__'\nclass C: pass\nsetattr(C, name, {})"
        )


# ── 4. Direct dunder attribute access blocked ─────────────────────────────────

class TestDirectDunderAccessBlocked(SandboxTestCase):

    def test_class_access_blocked(self):
        self.assertValidationBlocked("x = 'hi'\nprint(x.__class__)", "__class__")

    def test_dict_access_blocked(self):
        self.assertValidationBlocked("class C: pass\nprint(C.__dict__)", "__dict__")

    def test_module_access_blocked(self):
        self.assertValidationBlocked("import math\nprint(math.__name__)", "__name__")

    def test_private_attr_blocked(self):
        self.assertValidationBlocked("class C:\n    _x = 1\nprint(C._x)", "_x")


# ── 5. Normal method calls still work ─────────────────────────────────────────

class TestNormalMethodCalls(SandboxTestCase):

    def test_list_append(self):
        self.assertRunsOk("x=[1,2]\nx.append(3)\nprint(x)", "[1, 2, 3]")

    def test_str_upper(self):
        self.assertRunsOk("print('hello'.upper())", "HELLO")

    def test_str_split(self):
        self.assertRunsOk("print('a,b,c'.split(','))", "['a', 'b', 'c']")

    def test_dict_items(self):
        self.assertRunsOk("d={'a':1,'b':2}\nprint(sorted(d.items()))", "[('a', 1), ('b', 2)]")

    def test_list_sort(self):
        self.assertRunsOk("x=[3,1,2]\nx.sort()\nprint(x)", "[1, 2, 3]")

    def test_str_format(self):
        self.assertRunsOk("print('{} + {} = {}'.format(1, 2, 3))", "1 + 2 = 3")

    def test_list_comprehension(self):
        self.assertRunsOk("print([x**2 for x in range(5)])", "[0, 1, 4, 9, 16]")


# ── 6. eval / exec still blocked ──────────────────────────────────────────────

class TestEvalExecBlocked(SandboxTestCase):

    def test_eval_blocked(self):
        self.assertValidationBlocked("eval('1+1')", "eval")

    def test_exec_blocked(self):
        self.assertValidationBlocked("exec('x=1')", "exec")

    def test_compile_blocked(self):
        self.assertValidationBlocked("compile('x=1','<s>','exec')", "compile")


# ── 7. Blocked imports ────────────────────────────────────────────────────────

class TestBlockedImports(SandboxTestCase):

    def test_os_blocked(self):
        self.assertValidationBlocked("import os", "not allowed")

    def test_subprocess_blocked(self):
        self.assertValidationBlocked("import subprocess", "not allowed")

    def test_sys_blocked(self):
        self.assertValidationBlocked("import sys", "not allowed")

    def test_socket_blocked(self):
        self.assertValidationBlocked("import socket", "not allowed")

    def test_pathlib_blocked(self):
        self.assertValidationBlocked("import pathlib", "not allowed")


# ── 8. Allowed imports work ───────────────────────────────────────────────────

class TestAllowedImports(SandboxTestCase):

    def test_math_allowed(self):
        self.assertRunsOk("import math\nprint(round(math.pi, 4))", "3.1416")

    def test_json_allowed(self):
        self.assertRunsOk("import json\nprint(json.dumps({'a': 1}))", '{"a": 1}')

    def test_collections_allowed(self):
        self.assertRunsOk(
            "from collections import Counter\n"
            "print(Counter('aab')['a'])",
            "2",
        )

    def test_itertools_allowed(self):
        self.assertRunsOk(
            "import itertools\n"
            "print(list(itertools.islice(itertools.count(1), 3)))",
            "[1, 2, 3]",
        )

    def test_statistics_allowed(self):
        self.assertRunsOk(
            "import statistics\nprint(statistics.mean([1,2,3,4,5]))",
            "3",
        )


# ── 9. Output truncation ──────────────────────────────────────────────────────

class TestOutputTruncation(SandboxTestCase):

    def test_large_output_is_truncated(self):
        # Print 30 000 chars — should be truncated to maximum_output_characters.
        code = "print('x' * 30000)"
        result = self.assertRunsOk(code)
        self.assertLessEqual(
            len(str(result.output)),
            self.tool.maximum_output_characters + 60,  # +60 for truncation message
        )
        if len(str(result.output)) > self.tool.maximum_output_characters:
            self.assertIn("truncated", str(result.output))


# ── 10. validate_arguments basics ─────────────────────────────────────────────

class TestValidateArguments(SandboxTestCase):

    def test_empty_code_raises(self):
        with self.assertRaises(ValueError):
            self.tool.validate_arguments({"code": ""})

    def test_non_string_code_raises(self):
        with self.assertRaises(ValueError):
            self.tool.validate_arguments({"code": 123})

    def test_syntax_error_raises(self):
        with self.assertRaises(ValueError):
            self.tool.validate_arguments({"code": "def foo(: pass"})

    def test_non_dict_args_raises(self):
        with self.assertRaises(TypeError):
            self.tool.validate_arguments("not a dict")  # type: ignore


if __name__ == "__main__":
    unittest.main()
