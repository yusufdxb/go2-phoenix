"""Static bans: no LowCmd path sets kp=0, no Sport Damp call, no behaviour called DAMP.

Owner rule 2026-10-03: "DAMP should never be used whatsoever". The kp=0 scan is an AST
scan (comments and docstrings cannot trip it) over every src/ and scripts/ file under
``src/phoenix/sim2real`` or that mentions lowcmd. Whitelist for a folded terminal kp=0
state: empty, because the folded hold uses ``folded_kp`` > 0.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WHITELIST: set[str] = set()


def _is_zero(n: ast.AST) -> bool:
    """A literal zero in any spelling: 0, 0.0, -0.0, +0, float(0), float(), int(0), 0 * x, x * 0."""
    if isinstance(n, ast.Constant):
        return isinstance(n.value, int | float) and n.value == 0
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub | ast.UAdd):
        return _is_zero(n.operand)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("float", "int"):
        return not n.args and not n.keywords or (len(n.args) == 1 and _is_zero(n.args[0]))
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
        return _is_zero(n.left) or _is_zero(n.right)
    return False


def _zero_in(n: ast.AST | None) -> bool:
    if n is None:
        return False
    if _is_zero(n):
        return True
    if isinstance(n, ast.IfExp):
        return _zero_in(n.body) or _zero_in(n.orelse)
    return False


def _kp_name(n: ast.AST) -> bool:
    name = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else ""
    return name == "kp" or name.endswith("_kp")


def zero_kp_sites(source: str) -> list[int]:
    """Line numbers where the source assigns, defaults or passes a literal zero kp."""
    hits: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg and _kp_name(ast.Name(id=node.arg)):
            if _zero_in(node.value):
                hits.append(node.value.lineno)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if _kp_name(tgt) and _zero_in(node.value):
                    hits.append(node.lineno)
                if isinstance(tgt, ast.Tuple):
                    vals = node.value
                    elts = vals.elts if isinstance(vals, ast.Tuple) else []
                    if isinstance(vals, ast.IfExp):
                        elts = [e for b in (vals.body, vals.orelse) if isinstance(b, ast.Tuple) for e in b.elts[:1]]
                        elts = elts[:1]
                    for t_el, v_el in zip(tgt.elts, elts, strict=False):
                        if _kp_name(t_el) and _zero_in(v_el):
                            hits.append(node.lineno)
                    if isinstance(vals, ast.IfExp) and tgt.elts and _kp_name(tgt.elts[0]):
                        for b in (vals.body, vals.orelse):
                            if isinstance(b, ast.Tuple) and b.elts and _zero_in(b.elts[0]):
                                hits.append(node.lineno)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values, strict=False):
                if isinstance(k, ast.Constant) and k.value == "kp" and _zero_in(v):
                    hits.append(v.lineno)
        elif isinstance(node, ast.Call):
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fn in ("lowcmd_fields", "_publish") and len(node.args) >= 2 and _zero_in(node.args[1]):
                hits.append(node.lineno)
    return sorted(set(hits))


def _lowcmd_files() -> list[Path]:
    files = []
    for root in (REPO / "src", REPO / "scripts"):
        for p in root.rglob("*.py"):
            if "sim2real" in p.parts or re.search(r"lowcmd", p.read_text(), re.I):
                files.append(p)
    return files


def test_scanner_detects_the_patterns_it_bans() -> None:
    assert zero_kp_sites("x(kp=0.0)") == [1]
    assert zero_kp_sites("kp = 0") == [1]
    assert zero_kp_sites("kp, kd = (0.0, 1.0) if a else (1, 2)") == [1]
    assert zero_kp_sites("d = {'kp': 0}") == [1]
    assert zero_kp_sites("lowcmd_fields(t, 0.0, 1.0)") == [1]
    assert zero_kp_sites("# kp=0\n'''kp=0'''\nx(kp=5.0)") == []
    assert zero_kp_sites("kp = float(0)") == [1]
    assert zero_kp_sites("kp = float()") == [1]
    assert zero_kp_sites("x(kp=-0.0)") == [1]
    assert zero_kp_sites("hold_kp = 0 * gain") == [1]
    assert zero_kp_sites("kp = gain * 0.0") == [1]
    assert zero_kp_sites("d = {'kp': -0.0}") == [1]
    assert zero_kp_sites("lowcmd_fields(t, float(0), 1.0)") == [1]
    assert zero_kp_sites("kp = 5.0 * gain") == []
    assert api_1001_lines("d = {'api_id': 1001}") == [1]
    assert api_1001_lines('x = {"api_id":1001, "parameter": ""}') == [1]
    assert api_1001_lines("req = dict(api_id=1001)") == [1]
    # motion_switcher CheckMode is api 1001 on a different service: allowed where clearly so.
    assert api_1001_lines("d = {'api_id': 1001}  # motion_switcher CheckMode") == []


def test_no_lowcmd_path_sets_kp_zero() -> None:
    files = _lowcmd_files()
    assert len(files) > 5
    bad = {}
    for p in files:
        rel = str(p.relative_to(REPO))
        hits = zero_kp_sites(p.read_text())
        if hits and rel not in WHITELIST:
            bad[rel] = hits
    assert not bad, f"kp=0 on a LowCmd path: {bad}"


_API_1001 = re.compile(r"""["']?\bapi_id["']?\s*[:=]\s*1001\b""")
_SWITCHER = re.compile(r"motion_switcher|CheckMode", re.I)


def api_1001_lines(source: str) -> list[int]:
    """Lines that set api_id 1001 (Sport Damp) in any form, unless the line says it is
    the motion_switcher CheckMode call (also 1001, a different service)."""
    return [
        i
        for i, line in enumerate(source.splitlines(), 1)
        if _API_1001.search(line) and not _SWITCHER.search(line)
    ]


def test_no_sport_damp_call_or_api_1001_send() -> None:
    pat = re.compile(r"\bDamp\(|API_ID_DAMP|SPORT_API_ID_DAMP|api_id\s*=\s*1001|\.Damp\b")
    offenders = []
    for root in (REPO / "src", REPO / "scripts"):
        for p in list(root.rglob("*.py")) + list(root.rglob("*.sh")):
            text = p.read_text()
            if pat.search(text) or api_1001_lines(text):
                offenders.append(str(p.relative_to(REPO)))
    assert not offenders, offenders


def test_no_behaviour_is_called_damp_in_the_deploy_stack() -> None:
    pat = re.compile(r"\bdamp\b|damp_|_damp(?![a-z])|\bDAMP\b|\bDamp\b", re.I)
    files = list((REPO / "src/phoenix/sim2real").glob("*.py"))
    files += [REPO / "scripts/go2_release_sport_mode.py", REPO / "scripts/harness_preflight.sh"]
    files += [REPO / "scripts/dryrun_pipeline.sh"]
    offenders = [
        f"{p.relative_to(REPO)}:{i}"
        for p in files
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if pat.search(line)
    ]
    assert not offenders, offenders
