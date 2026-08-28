"""Scan loop files for the four gate-machinery defects gate_guard.Gates now prevents.

Written instead of blanket-retrofitting every loop. Most loops that already ran are correct as
they stand -- each defect was patched where it appeared -- and rewriting a file that produced a
committed result risks changing it silently, which is worse than leaving it alone. This finds the
files where the defect is still LATENT, so the retrofit is aimed rather than swept.

  B  a verdict message that indexes something which may not exist on the branch not taken.
     GG.verdict builds BOTH f-strings before the call, so `if_true=f"...{best[0]}..."` crashes
     when best is empty and the gate had already decided FAIL. Loop 196's X4 and loop 197's Y4.
  C  a hand-rolled void set, which is where loop 196's X4 printed VOID and FAIL into one summary.
  A  a gate whose verdict is a threshold comparison on a value that could be nan -- the boolean
     swallows the undefinedness, which is loop 187's B6.
  D  a control gate comparing signed values, which assumes the sign of its own answer -- loop
     199's Q5.

AND TWO THAT ARE NOT GATE DEFECTS BUT KILL A RUN THE SAME WAY, ADDED AFTER LOOP 241.

  E  `requires="Z2"` -- a bare string where a tuple was meant. `requires` is consumed by
     iteration, so a string iterates as its CHARACTERS, none of which is a registered gate, and
     the gate is VOID whatever its precondition did. gate_guard now wraps a string rather than
     iterating it, so this no longer breaks a run; it is still flagged because the tuple form
     says what was meant.

  F  an imported module name rebound inside a function. Loop 241 wrote `nn = tr10[...]` inside
     main() for a nearest-neighbour index, which made `nn` a LOCAL of main and therefore unbound
     for every nested function that closed over it -- so `class PairNet(nn.Module)` died on
     UnboundLocalError before a single number was computed, forty lines above the assignment that
     caused it. Not a gate defect. Caught here because it is mechanical, cheap, and the error it
     produces points at the wrong line.

Run: python3 colab/lint_gates.py
"""
import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RISKY_INDEX = re.compile(r"if_true\s*=\s*f?\"[^\"]*\{[^}]*\[[^\]]+\][^}]*\}")
HAND_VOID = re.compile(r"^\s*void\s*(=|\|=)\s*", re.M)
NAN_CMP = re.compile(r"np\.isfinite\(\s*(\w+)\s*\)\s*and\s*\1\s*[<>]")
SIGNED_CTRL = re.compile(r"\b(\w*real\w*|d\d)\[?[\"']?\w*[\"']?\]?\s*>\s*(\w*swap\w*|\w*perm\w*|d\d)\b",
                         re.I)


# G: verbs a NEGATIVE value contradicts, inside an if_false whose value prints its own sign.
# "adds" is deliberately NOT here: "adds -0.15" is coherent arithmetic, so flagging it would
# fire on a third of the corpus and train me to ignore the linter. "costs -0.05" is the bug --
# it means the thing GAINED. Same for beats/exceeds/improves, which assert superiority.
# G: a directional verb in an if_false message, applied to a value printed with its own
# sign. Two regex versions of this check were WRONG in the same way -- a text window after
# `if_false=` ran past the end of the message and matched a verb from a neighbouring say()
# against a value from a different gate. Regexing over raw source cannot see where an
# argument ends, so this walks the AST and inspects ONLY the if_false expression.
# "adds" is deliberately absent: "adds -0.15" is coherent arithmetic and flagging it fired
# on a third of the corpus, which trains the reader to ignore the linter.
DIRECTIONAL_VERBS = re.compile(
    r"\b(costs?|loses?|lost|drops?|falls?|gains?|improves?|beats?|exceeds?|outperforms?|"
    r"rises?)\b", re.I)


def _signed_anywhere(node):
    for n in ast.walk(node):
        if isinstance(n, ast.FormattedValue) and n.format_spec is not None:
            for c in ast.walk(n.format_spec):
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and "+" in c.value:
                    return True
    return False


def _unguarded_text(node, out):
    """Literal text NOT inside a conditional branch.

    The prescribed fix for defect G is to READ the sign before using a directional verb:
        f"... is worth {d:+.4f}" + (f"... zeroing it IMPROVES it by {-d:.4f}" if d < 0
                                    else f"... below the bar")
    A verb inside an IfExp branch is therefore guarded by construction, and flagging it
    would train the author away from the very pattern the ledger prescribes. Only verbs in
    the UNCONDITIONAL part of the message are a coin flip on the sign."""
    if isinstance(node, ast.IfExp):
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    for ch in ast.iter_child_nodes(node):
        _unguarded_text(ch, out)


def _fstring_parts(node):
    """(unconditional literal text, whether any interpolation prints an explicit sign)."""
    out = []
    _unguarded_text(node, out)
    return " ".join(out), _signed_anywhere(node)


def sign_assuming_fail_messages(src):
    """Defect G, by AST rather than by text window."""
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add"):
            continue
        for kw in node.keywords:
            if kw.arg != "if_false":
                continue
            body = kw.value.body if isinstance(kw.value, ast.Lambda) else kw.value
            text, signed = _fstring_parts(body)
            m = DIRECTIONAL_VERBS.search(text)
            if m and signed:
                out.append(f"'{m.group(1)}' with a signed value: {text[:64]}")
    return out


def scan(p):
    s = p.read_text()
    if "gate_guard" not in s or "def main" not in s:
        return None
    hits = {}
    # ONLY flag an eager message whose indexed NAME is assigned None somewhere in the file.
    # The first version flagged every f-string containing a subscript, which is most of them and
    # almost all benign -- a dict that always exists indexes fine. The dangerous shape is
    # specifically `d = None` followed by `if_true=f"...{d['k']}..."`, because Python builds that
    # string at the CALL SITE before verdict() can intercept it. That is exactly loop 196's X4
    # (winner = None) and loop 197's Y4 (qual = []). An over-flagging linter gets ignored, which
    # is worse than no linter.
    nullable = set(re.findall(r"^\s*(\w+)\s*=\s*(?:None|False,\s*None|\[\])\s*$", s, re.M))
    nullable |= set(re.findall(r"^\s*\w+,\s*(\w+)\s*=\s*\w+,\s*None\s*$", s, re.M))
    b = []
    for m in RISKY_INDEX.finditer(s):
        txt = m.group(0)
        names = set(re.findall(r"\{(\w+)\[", txt))
        if names & nullable:
            b.append(f"{sorted(names & nullable)} may be None: {txt[:70]}")
    if b:
        hits["B eager message indexes a name assigned None"] = b
    if HAND_VOID.search(s) and "Gates(" not in s:
        hits["C hand-rolled void set"] = [f"{len(HAND_VOID.findall(s))} sites"]
    # A: a gate guarded by isfinite is SAFE; flag comparisons with no such guard on a z/rho name
    unguarded = [m.group(0)[:70] for m in
                 re.finditer(r"^\s*\w+ = bool\((?!.*isfinite)[^)]*\b(z\w*|rho|r_)\b[^)]*[<>][^)]*\)",
                             s, re.M)]
    if unguarded:
        hits["A threshold on a possibly-undefined statistic, unguarded"] = unguarded
    c = [m.group(0)[:70] for m in SIGNED_CTRL.finditer(s)]
    if c:
        hits["D control compared by sign, not magnitude"] = c
    e = [m.group(0) for m in re.finditer(r'requires\s*=\s*"[A-Za-z0-9_]+"', s)]
    if e:
        hits["E requires= given a bare string where a tuple was meant"] = sorted(set(e))
    f = shadowed_imports(s)
    if f:
        hits["F an imported name is rebound inside a function"] = f
    # G: a directional verb inside an if_false message, applied to a value printed with an
    # explicit sign. A FAIL branch is where the statistic is LEAST constrained in sign, so
    # "costs {d:+.4f}" renders as "costs -0.0543" -- the verb and the sign disagree. Only
    # if_false is flagged: an if_true branch has usually already established the direction.
    g = sign_assuming_fail_messages(s)
    if g:
        hits["G if_false message uses a directional verb on a signed statistic"] = sorted(set(g))
    return hits


def shadowed_imports(src):
    """Every module-level import name that some function also assigns to.

    Python decides local-vs-global per function at COMPILE time, so a single assignment anywhere
    in a function body makes that name local for the WHOLE body -- and for every nested function
    closing over it. Loop 241's `nn` is the worked example."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imported.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                imported.add(a.asname or a.name)
    out = []
    for fn in [x for x in ast.walk(tree)
               if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for t in ast.walk(fn):
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store) and t.id in imported:
                out.append(f"{t.id} rebound in {fn.name}() at line {t.lineno}")
    return sorted(set(out))


def main():
    files = sorted(HERE.glob("loop_*.py"))
    flagged = 0
    for p in files:
        h = scan(p)
        if not h:
            continue
        if any(h.values()):
            flagged += 1
            print(f"\n{p.name}")
            for k, v in h.items():
                print(f"   {k}")
                for x in v[:3]:
                    print(f"      {x}")
    print(f"\n{flagged} of {len(files)} loop files carry a latent gate defect")
    return flagged


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
