from __future__ import annotations

from dataclasses import dataclass, field
import ast
import re


STRICT_PATTERNS = {
    "code_bypass": r"\btry\s*:\s*",
    "timing_patch": r"(torch\.cuda\.Event|time\.perf_counter|time\.time)\s*=",
    "threading": r"\b(threading|multiprocessing|concurrent\.futures)\b",
    "lazy_tensor": r"(_make_subclass|class\s+\w+\(.*torch\.Tensor.*\))",
    "pass_statement": r"(^|\n)\s*pass\b",
}

WARNING_PATTERNS = {
    "streams": r"(torch\.cuda\.Stream|with\s+torch\.cuda\.stream|\.wait_stream\s*\()",
}

PTX_FORBIDDEN = {
    "cuda_inline": r"(load_inline|cpp_extension|cuda_sources)",
    "triton": r"(@triton\.jit|\btl\.)",
    "cutlass": r"(cute::|cutlass::)",
    "torch_compute": r"\b(torch\.(matmul|mm|bmm|relu|gelu|softmax|conv\d*d?|layer_norm|batch_norm)|F\.)",
}

PTX_REQUIRED = {
    "modelnew": r"\bclass\s+ModelNew\b",
    "ptx_sources": r"\bPTX_SOURCES\b",
    "ptx_kernels": r"\bPTX_KERNELS\b",
    "ptx_spec": r"\bPTXKernelSpec\b",
}

CUDA_REQUIRED = {
    "modelnew": r"\bclass\s+ModelNew\b",
    "kernel_impl": r"(__global__|load_inline|cpp_extension)",
}

ALLOWED_IMPORTS = {
    ("import", "torch", None),
    ("import", "torch.nn", "nn"),
    ("from", "ptxbench.runtime", "PTXModuleRunner"),
    ("from", "ptxbench.spec", "PTXKernelSpec"),
}

FORBIDDEN_TORCH_FUNCTIONS = {
    "torch.add",
    "torch.matmul",
    "torch.mm",
    "torch.bmm",
    "torch.relu",
    "torch.gelu",
    "torch.softmax",
    "torch.sum",
    "torch.mean",
    "torch.max",
    "torch.min",
}

FORBIDDEN_TENSOR_METHODS = {
    "sum",
    "mean",
    "relu",
    "matmul",
    "max",
    "min",
}

ALLOWED_TORCH_ALLOCATORS = {
    "torch.empty",
    "torch.empty_like",
    "torch.empty_strided",
}

ALLOWED_TENSOR_METADATA_CALLS = {
    "numel",
    "size",
    "stride",
    "is_contiguous",
}

ALLOWED_TENSOR_METADATA_ATTRS = {
    "shape",
    "device",
    "dtype",
}

ALLOWED_SCALAR_BUILTINS = {
    "abs",
    "bool",
    "float",
    "int",
    "len",
    "max",
    "min",
    "round",
}

SCALAR_ANNOTATIONS = {
    "bool",
    "builtins.bool",
    "float",
    "builtins.float",
    "int",
    "builtins.int",
}

FORBIDDEN_NAME_ERRORS = {
    "open": "ptx_forbidden:open",
    "eval": "ptx_forbidden:eval",
    "exec": "ptx_forbidden:exec",
    "compile": "ptx_forbidden:compile",
    "__import__": "ptx_forbidden:__import__",
    "globals": "ptx_forbidden:globals",
    "locals": "ptx_forbidden:locals",
    "vars": "ptx_forbidden:vars",
    "getattr": "ptx_forbidden:getattr",
    "setattr": "ptx_forbidden:setattr",
    "os": "ptx_forbidden:os",
    "sys": "ptx_forbidden:sys",
    "pathlib": "ptx_forbidden:pathlib",
    "importlib": "ptx_forbidden:importlib",
    "inspect": "ptx_forbidden:inspect",
    "socket": "ptx_forbidden:socket",
    "requests": "ptx_forbidden:requests",
    "urllib": "ptx_forbidden:urllib",
}


@dataclass
class StaticCheckResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __iter__(self):
        yield self.valid
        yield self.errors
        yield self.warnings


CHECK_MESSAGES = {
    "code_bypass": "strict:try_except",
    "timing_patch": "strict:timing_patch",
    "threading": "strict:threading",
    "lazy_tensor": "strict:lazy_tensor",
    "pass_statement": "strict:pass_statement",
    "streams": "warning:streams",
    "precision_downgrade": "warning:precision_downgrade",
}

DEFAULT_FORBIDDEN = frozenset(STRICT_PATTERNS)
DEFAULT_WARNINGS = frozenset({"streams", "precision_downgrade"})

PRECISION_DOWNGRADE_PATTERN = re.compile(
    r"(\.half\s*\(\s*\)|torch\.(half|float16)|dtype\s*=\s*torch\.(half|float16))"
)


def _normalize_precision(precision: str | None) -> str:
    normalized = (precision or "fp32").lower()
    aliases = {
        "float32": "fp32",
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
    }
    return aliases.get(normalized, normalized)


def _append_check(
    name: str,
    *,
    errors: list[str],
    warnings: list[str],
    forbidden_checks: set[str],
    warning_checks: set[str],
) -> None:
    message = CHECK_MESSAGES[name]
    if name in forbidden_checks:
        errors.append(message)
        return
    if name in warning_checks:
        warnings.append(message)


def _dotted_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return None


def _annotation_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return None


def _is_tensor_annotation(node: ast.AST | None) -> bool:
    annotation = (_annotation_name(node) or "").lower()
    return "tensor" in annotation


def _module_error_name(module_name: str) -> str:
    if module_name.startswith("torch.utils.cpp_extension") or "cpp_extension" in module_name:
        return "ptx_forbidden:cuda_inline"
    if module_name == "subprocess" or module_name.startswith("subprocess."):
        return "ptx_forbidden:subprocess"
    if module_name == "ctypes" or module_name.startswith("ctypes."):
        return "ptx_forbidden:ctypes"
    if module_name == "multiprocessing" or module_name.startswith("multiprocessing."):
        return "ptx_forbidden:threading"
    if module_name == "threading" or module_name.startswith("threading."):
        return "ptx_forbidden:threading"
    if module_name == "concurrent.futures" or module_name.startswith("concurrent.futures."):
        return "ptx_forbidden:threading"
    if module_name == "triton" or module_name.startswith("triton."):
        return "ptx_forbidden:triton"
    if module_name == "cupy" or module_name.startswith("cupy."):
        return "ptx_forbidden:cupy"
    if module_name == "numba" or module_name.startswith("numba."):
        return "ptx_forbidden:numba"
    if module_name == "time" or module_name.startswith("time."):
        return "ptx_forbidden:timing_hack"
    for name, error in FORBIDDEN_NAME_ERRORS.items():
        if module_name == name or module_name.startswith(f"{name}."):
            return error
    return "ptx_forbidden:disallowed_import"


def _add_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def _check_allowed_imports(tree: ast.AST, errors: list[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                key = ("import", alias.name, alias.asname)
                if key not in ALLOWED_IMPORTS:
                    _add_error(errors, _module_error_name(alias.name))
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if len(node.names) != 1 or node.level != 0:
                _add_error(errors, _module_error_name(module_name))
                continue
            imported_name = node.names[0].name
            key = ("from", module_name, imported_name)
            if key not in ALLOWED_IMPORTS:
                combined = module_name or imported_name
                _add_error(errors, _module_error_name(combined))


def _is_literal_string(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _is_literal_string_dict(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    return all(
        _is_literal_string(key) and _is_literal_string(value)
        for key, value in zip(node.keys, node.values, strict=True)
    )


def _check_literal_ptx_sources(tree: ast.AST, errors: list[str]) -> None:
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "PTX_SOURCES" for target in node.targets):
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "PTX_SOURCES":
                value = node.value
        if value is None:
            continue
        if not (_is_literal_string(value) or _is_literal_string_dict(value)):
            _add_error(errors, "ptx_forbidden:literal_ptx_sources")


def _check_global_ptx_forbidden_nodes(tree: ast.AST, errors: list[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            _add_error(errors, "ptx_forbidden:try_except")
            continue

        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAME_ERRORS:
            _add_error(errors, FORBIDDEN_NAME_ERRORS[node.id])
            continue

        if isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            if dotted:
                root = dotted.split(".", 1)[0]
                if root in FORBIDDEN_NAME_ERRORS:
                    _add_error(errors, FORBIDDEN_NAME_ERRORS[root])
                if dotted.startswith("torch.ops"):
                    _add_error(errors, "ptx_forbidden:torch_ops")
                if dotted.startswith("torch.cuda.Stream"):
                    _add_error(errors, "ptx_forbidden:stream_hack")
            continue

        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func) or ""
            if dotted:
                root = dotted.split(".", 1)[0]
                if root in FORBIDDEN_NAME_ERRORS:
                    _add_error(errors, FORBIDDEN_NAME_ERRORS[root])
            if dotted.startswith("torch.ops"):
                _add_error(errors, "ptx_forbidden:torch_ops")
            if dotted == "torch.compile":
                _add_error(errors, "ptx_forbidden:torch_compile")
            if dotted in FORBIDDEN_TORCH_FUNCTIONS or dotted.startswith("torch.conv"):
                _add_error(errors, "ptx_forbidden:torch_compute")
            if "load_inline" in dotted or "cpp_extension" in dotted:
                _add_error(errors, "ptx_forbidden:cuda_inline")
            if dotted.startswith("subprocess."):
                _add_error(errors, "ptx_forbidden:subprocess")
            if dotted.startswith("ctypes."):
                _add_error(errors, "ptx_forbidden:ctypes")
            if dotted.startswith(("threading.", "multiprocessing.", "concurrent.futures.")):
                _add_error(errors, "ptx_forbidden:threading")
            if dotted in {"time.time", "time.perf_counter"} or dotted.startswith("torch.cuda.Event"):
                _add_error(errors, "ptx_forbidden:timing_hack")
            if dotted.startswith(("torch.cuda.Stream", "torch.cuda.stream", "torch.cuda.synchronize")):
                _add_error(errors, "ptx_forbidden:stream_hack")


class _ForwardValidator:
    def __init__(self, function: ast.FunctionDef):
        self.function = function
        self.errors: list[str] = []
        self.env: dict[str, str] = {"self": "self"}
        self.launch_tensor_names: set[str] = set()
        self.launch_call_count = 0
        for arg in function.args.args[1:]:
            annotation_name = _annotation_name(arg.annotation)
            if _is_tensor_annotation(arg.annotation) or annotation_name is None:
                self.env[arg.arg] = "tensor_input"
            elif annotation_name in SCALAR_ANNOTATIONS:
                self.env[arg.arg] = "scalar"
            else:
                self.env[arg.arg] = "scalar"

    def add_error(self, message: str) -> None:
        _add_error(self.errors, message)

    def validate(self) -> list[str]:
        for statement in self.function.body:
            self._validate_statement(statement)
        if self.launch_call_count == 0:
            self.add_error("ptx_required:launch_call")
        return self.errors

    def _is_tensor_kind(self, kind: str) -> bool:
        return kind in {"tensor_input", "tensor_output"}

    def _is_scalar_kind(self, kind: str) -> bool:
        return kind in {"scalar", "bool"}

    def _validate_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Assign):
            value_kind = self._infer_expr(statement.value)
            if value_kind == "unknown":
                self.add_error("ptx_forbidden:disallowed_expr")
            for target in statement.targets:
                self._bind_target(target, value_kind)
            return
        if isinstance(statement, ast.AnnAssign):
            value_kind = self._infer_expr(statement.value) if statement.value is not None else "unknown"
            if value_kind == "unknown":
                self.add_error("ptx_forbidden:disallowed_expr")
            self._bind_target(statement.target, value_kind)
            return
        if isinstance(statement, ast.Expr):
            value_kind = self._infer_expr(statement.value)
            if value_kind == "unknown":
                self.add_error("ptx_forbidden:disallowed_expr")
            return
        if isinstance(statement, ast.Return):
            self._validate_return(statement.value)
            return
        if isinstance(statement, ast.If):
            condition_kind = self._infer_expr(statement.test)
            if not self._is_scalar_kind(condition_kind):
                self.add_error("ptx_forbidden:disallowed_control_flow")
            for branch_statement in statement.body:
                self._validate_statement(branch_statement)
            for branch_statement in statement.orelse:
                self._validate_statement(branch_statement)
            return
        if isinstance(statement, ast.Raise):
            if statement.exc is not None:
                self._infer_expr(statement.exc)
            return
        self.add_error("ptx_forbidden:disallowed_control_flow")

    def _bind_target(self, target: ast.expr, kind: str) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = kind
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            child_kind = "scalar" if kind == "shape" else kind
            for item in target.elts:
                self._bind_target(item, child_kind)
            return
        if isinstance(target, ast.Attribute):
            dotted = _dotted_name(target)
            if dotted == "self.runner" and kind == "runner":
                return
            self.add_error("ptx_forbidden:disallowed_state")
            return
        self.add_error("ptx_forbidden:disallowed_state")

    def _validate_return(self, value: ast.expr | None) -> None:
        if value is None:
            return
        if isinstance(value, (ast.Tuple, ast.List)):
            for item in value.elts:
                self._validate_return(item)
            return
        if isinstance(value, ast.Dict):
            for item in value.values:
                if item is not None:
                    self._validate_return(item)
            return

        kind = self._infer_expr(value)
        if kind == "tensor_input":
            self.add_error("ptx_forbidden:return_input_tensor")
            return
        if kind == "tensor_output":
            if isinstance(value, ast.Name) and value.id in self.launch_tensor_names:
                return
            self.add_error("ptx_forbidden:return_unlaunched_output")
            return
        if self._is_tensor_kind(kind) or kind in {"runner", "self", "unknown"}:
            self.add_error("ptx_forbidden:disallowed_return")

    def _infer_expr(self, expr: ast.expr | None) -> str:
        if expr is None:
            return "none"
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, bool):
                return "bool"
            if isinstance(expr.value, (int, float, str)) or expr.value is None:
                return "scalar"
            return "unknown"
        if isinstance(expr, ast.Name):
            if expr.id in FORBIDDEN_NAME_ERRORS:
                self.add_error(FORBIDDEN_NAME_ERRORS[expr.id])
                return "unknown"
            return self.env.get(expr.id, "unknown")
        if isinstance(expr, ast.Attribute):
            dotted = _dotted_name(expr)
            if dotted == "self.runner":
                return "runner"
            if dotted:
                root = dotted.split(".", 1)[0]
                if root in FORBIDDEN_NAME_ERRORS:
                    self.add_error(FORBIDDEN_NAME_ERRORS[root])
                    return "unknown"
                if dotted.startswith("torch.ops"):
                    self.add_error("ptx_forbidden:torch_ops")
                    return "unknown"
            base_kind = self._infer_expr(expr.value)
            if self._is_tensor_kind(base_kind) and expr.attr in ALLOWED_TENSOR_METADATA_ATTRS:
                if expr.attr == "shape":
                    return "shape"
                return "scalar"
            return "unknown"
        if isinstance(expr, ast.Subscript):
            base_kind = self._infer_expr(expr.value)
            if base_kind == "shape":
                self._infer_expr(expr.slice)
                return "scalar"
            return "unknown"
        if isinstance(expr, (ast.Tuple, ast.List)):
            element_kinds = [self._infer_expr(item) for item in expr.elts]
            if all(kind in {"scalar", "bool", "shape"} for kind in element_kinds):
                return "shape"
            return "unknown"
        if isinstance(expr, ast.Dict):
            for key in expr.keys:
                if key is not None:
                    self._infer_expr(key)
            for value in expr.values:
                self._infer_expr(value)
            return "unknown"
        if isinstance(expr, ast.UnaryOp):
            operand_kind = self._infer_expr(expr.operand)
            if self._is_tensor_kind(operand_kind):
                self.add_error("ptx_forbidden:tensor_binop")
                return "unknown"
            return "bool" if isinstance(expr.op, ast.Not) else operand_kind
        if isinstance(expr, ast.BinOp):
            left_kind = self._infer_expr(expr.left)
            right_kind = self._infer_expr(expr.right)
            if isinstance(expr.op, ast.MatMult) or self._is_tensor_kind(left_kind) or self._is_tensor_kind(right_kind):
                self.add_error("ptx_forbidden:tensor_binop")
                return "unknown"
            if self._is_scalar_kind(left_kind) and self._is_scalar_kind(right_kind):
                return "scalar"
            return "unknown"
        if isinstance(expr, ast.BoolOp):
            operand_kinds = [self._infer_expr(value) for value in expr.values]
            if all(self._is_scalar_kind(kind) for kind in operand_kinds):
                return "bool"
            return "unknown"
        if isinstance(expr, ast.Compare):
            left_kind = self._infer_expr(expr.left)
            comparator_kinds = [self._infer_expr(value) for value in expr.comparators]
            if self._is_scalar_kind(left_kind) and all(self._is_scalar_kind(kind) for kind in comparator_kinds):
                return "bool"
            return "unknown"
        if isinstance(expr, ast.Call):
            return self._infer_call(expr)
        if isinstance(expr, ast.IfExp):
            test_kind = self._infer_expr(expr.test)
            body_kind = self._infer_expr(expr.body)
            else_kind = self._infer_expr(expr.orelse)
            if self._is_scalar_kind(test_kind) and body_kind == else_kind:
                return body_kind
            return "unknown"
        return "unknown"

    def _infer_call(self, call: ast.Call) -> str:
        dotted = _dotted_name(call.func) or ""
        if dotted:
            root = dotted.split(".", 1)[0]
            if root in FORBIDDEN_NAME_ERRORS:
                self.add_error(FORBIDDEN_NAME_ERRORS[root])
                return "unknown"
        if dotted.startswith("torch.ops"):
            self.add_error("ptx_forbidden:torch_ops")
            return "unknown"
        if dotted == "torch.compile":
            self.add_error("ptx_forbidden:torch_compile")
            return "unknown"
        if dotted in FORBIDDEN_TORCH_FUNCTIONS or dotted.startswith("torch.conv"):
            self.add_error("ptx_forbidden:torch_compute")
            return "unknown"
        if "load_inline" in dotted or "cpp_extension" in dotted:
            self.add_error("ptx_forbidden:cuda_inline")
            return "unknown"
        if dotted.startswith("subprocess."):
            self.add_error("ptx_forbidden:subprocess")
            return "unknown"
        if dotted.startswith("ctypes."):
            self.add_error("ptx_forbidden:ctypes")
            return "unknown"
        if dotted.startswith(("threading.", "multiprocessing.", "concurrent.futures.")):
            self.add_error("ptx_forbidden:threading")
            return "unknown"
        if dotted in {"time.time", "time.perf_counter"} or dotted.startswith("torch.cuda.Event"):
            self.add_error("ptx_forbidden:timing_hack")
            return "unknown"
        if dotted.startswith(("torch.cuda.Stream", "torch.cuda.stream", "torch.cuda.synchronize")):
            self.add_error("ptx_forbidden:stream_hack")
            return "unknown"
        if dotted in ALLOWED_TORCH_ALLOCATORS:
            return self._infer_allocator_call(dotted, call)
        if dotted == "self.runner.launch":
            return self._infer_launch_call(call)
        if dotted == "PTXModuleRunner":
            for argument in call.args:
                self._infer_expr(argument)
            for keyword in call.keywords:
                if keyword.value is not None:
                    self._infer_expr(keyword.value)
            return "runner"
        if dotted == "PTXKernelSpec":
            for argument in call.args:
                self._infer_expr(argument)
            for keyword in call.keywords:
                if keyword.value is not None:
                    self._infer_expr(keyword.value)
            return "scalar"
        if dotted == "super":
            return "scalar"
        if dotted == "super().__init__":
            for argument in call.args:
                self._infer_expr(argument)
            for keyword in call.keywords:
                if keyword.value is not None:
                    self._infer_expr(keyword.value)
            return "scalar"
        if dotted in ALLOWED_SCALAR_BUILTINS:
            argument_kinds = [self._infer_expr(argument) for argument in call.args]
            if dotted == "len" and argument_kinds == ["shape"]:
                return "scalar"
            if argument_kinds and not all(kind in {"scalar", "bool", "shape"} for kind in argument_kinds):
                return "unknown"
            return "scalar"
        if isinstance(call.func, ast.Attribute):
            base_kind = self._infer_expr(call.func.value)
            if self._is_tensor_kind(base_kind):
                if call.func.attr in ALLOWED_TENSOR_METADATA_CALLS:
                    return self._infer_tensor_metadata_call(call.func.attr, call)
                if call.func.attr in FORBIDDEN_TENSOR_METHODS:
                    self.add_error("ptx_forbidden:tensor_compute")
                    return "unknown"
                self.add_error("ptx_forbidden:disallowed_tensor_method")
                return "unknown"
        if dotted.startswith("torch."):
            self.add_error("ptx_forbidden:disallowed_torch_call")
            return "unknown"
        self.add_error("ptx_forbidden:disallowed_call")
        return "unknown"

    def _infer_allocator_call(self, dotted: str, call: ast.Call) -> str:
        if dotted == "torch.empty":
            for argument in call.args:
                argument_kind = self._infer_expr(argument)
                if argument_kind not in {"scalar", "shape"}:
                    self.add_error("ptx_forbidden:disallowed_torch_call")
            for keyword in call.keywords:
                if keyword.value is not None:
                    keyword_kind = self._infer_expr(keyword.value)
                    if self._is_tensor_kind(keyword_kind):
                        self.add_error("ptx_forbidden:disallowed_torch_call")
            return "tensor_output"
        if dotted == "torch.empty_like":
            if not call.args:
                self.add_error("ptx_forbidden:disallowed_torch_call")
                return "unknown"
            if not self._is_tensor_kind(self._infer_expr(call.args[0])):
                self.add_error("ptx_forbidden:disallowed_torch_call")
            for keyword in call.keywords:
                if keyword.value is not None:
                    self._infer_expr(keyword.value)
            return "tensor_output"
        if dotted == "torch.empty_strided":
            if len(call.args) < 2:
                self.add_error("ptx_forbidden:disallowed_torch_call")
                return "unknown"
            shape_kind = self._infer_expr(call.args[0])
            stride_kind = self._infer_expr(call.args[1])
            if shape_kind != "shape" or stride_kind != "shape":
                self.add_error("ptx_forbidden:disallowed_torch_call")
            for keyword in call.keywords:
                if keyword.value is not None:
                    self._infer_expr(keyword.value)
            return "tensor_output"
        return "unknown"

    def _infer_launch_call(self, call: ast.Call) -> str:
        if self._infer_expr(call.func.value) != "runner":
            self.add_error("ptx_forbidden:disallowed_call")
            return "unknown"
        self.launch_call_count += 1
        for argument in call.args:
            argument_kind = self._infer_expr(argument)
            if isinstance(argument, ast.Name) and self._is_tensor_kind(argument_kind):
                self.launch_tensor_names.add(argument.id)
        for keyword in call.keywords:
            if keyword.value is None:
                continue
            keyword_kind = self._infer_expr(keyword.value)
            if isinstance(keyword.value, ast.Name) and self._is_tensor_kind(keyword_kind):
                self.launch_tensor_names.add(keyword.value.id)
        return "scalar"

    def _infer_tensor_metadata_call(self, attr: str, call: ast.Call) -> str:
        for argument in call.args:
            argument_kind = self._infer_expr(argument)
            if not self._is_scalar_kind(argument_kind):
                self.add_error("ptx_forbidden:disallowed_tensor_method")
                return "unknown"
        for keyword in call.keywords:
            if keyword.value is not None:
                keyword_kind = self._infer_expr(keyword.value)
                if not self._is_scalar_kind(keyword_kind):
                    self.add_error("ptx_forbidden:disallowed_tensor_method")
                    return "unknown"
        if attr == "is_contiguous":
            return "bool"
        if attr in {"size", "stride"} and not call.args and not call.keywords:
            return "shape"
        return "scalar"


def _find_modelnew_class(tree: ast.AST) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            return node
    return None


def _find_class_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef | None:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


def _validate_ptx_ast(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["ptx_static:syntax_error"]

    errors: list[str] = []
    _check_allowed_imports(tree, errors)
    _check_global_ptx_forbidden_nodes(tree, errors)
    _check_literal_ptx_sources(tree, errors)

    modelnew = _find_modelnew_class(tree)
    if modelnew is not None:
        forward = _find_class_method(modelnew, "forward")
        if forward is None:
            _add_error(errors, "ptx_required:forward")
        else:
            validator = _ForwardValidator(forward)
            for error in validator.validate():
                _add_error(errors, error)

    return errors


def validate_submission_static(
    source: str,
    backend: str | None = None,
    *,
    precision: str = "fp32",
    forbidden: list[str] | None = None,
    warnings: list[str] | None = None,
) -> StaticCheckResult:
    backend = backend.lower() if backend else None
    errors: list[str] = []
    warning_messages: list[str] = []
    forbidden_checks = set(DEFAULT_FORBIDDEN if forbidden is None else forbidden)
    warning_checks = set(DEFAULT_WARNINGS if warnings is None else warnings)

    for label, pattern in STRICT_PATTERNS.items():
        if re.search(pattern, source):
            _append_check(
                label,
                errors=errors,
                warnings=warning_messages,
                forbidden_checks=forbidden_checks,
                warning_checks=warning_checks,
            )

    for label, pattern in WARNING_PATTERNS.items():
        if re.search(pattern, source):
            _append_check(
                label,
                errors=errors,
                warnings=warning_messages,
                forbidden_checks=forbidden_checks,
                warning_checks=warning_checks,
            )

    if _normalize_precision(precision) == "fp32" and PRECISION_DOWNGRADE_PATTERN.search(source):
        _append_check(
            "precision_downgrade",
            errors=errors,
            warnings=warning_messages,
            forbidden_checks=forbidden_checks,
            warning_checks=warning_checks,
        )

    if backend == "ptx":
        for label, pattern in PTX_FORBIDDEN.items():
            if re.search(pattern, source):
                _add_error(errors, f"ptx_forbidden:{label}")
        for label, pattern in PTX_REQUIRED.items():
            if not re.search(pattern, source):
                _add_error(errors, f"ptx_required:{label}")
        for error in _validate_ptx_ast(source):
            _add_error(errors, error)
    elif backend == "cuda":
        for label, pattern in CUDA_REQUIRED.items():
            if not re.search(pattern, source):
                _add_error(errors, f"cuda_required:{label}")
    elif backend is not None:
        _add_error(errors, f"unsupported_backend:{backend}")

    return StaticCheckResult(valid=not errors, errors=errors, warnings=warning_messages)
