"""GUM uncertainty propagation for one explicit measurement model.

The calculation follows the first-order law of propagation and the
Welch–Satterthwaite formula. A Monte Carlo option draws the same
uncorrelated inputs and evaluates the model on those draws.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import sympy
from scipy import stats
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

# No implicit multiplication: adjacent letters are one quantity (ab, ID_first).
# evaluate=False emits Mul, Add and Pow, and keeps the entered factor order.
_TRANSFORMS = standard_transformations + (convert_xor,)

# SymPy's default namespace treats I as the imaginary unit and E, N, beta and
# gamma as special objects. Those names are ordinary quantities here.
_GLOBAL = {
    "Symbol": sympy.Symbol,
    "Integer": sympy.Integer,
    "Float": sympy.Float,
    "Rational": sympy.Rational,
    "Add": sympy.Add,
    "Mul": sympy.Mul,
    "Pow": sympy.Pow,
}

_LOCALS = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "sqrt": sympy.sqrt,
    "log": sympy.log,
    "ln": sympy.log,
    "exp": sympy.exp,
    "abs": sympy.Abs,
    "pi": sympy.pi,
    "e": sympy.E,
}
_CONSTANTS = frozenset({"pi", "e"})

_DECIMAL_COMMA = re.compile(r"(\d),(\d)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_NAME = re.compile(r"^[^\d\W]\w*$", re.UNICODE)
_IDENT = r"[^\d\W]\w*"
_SCIENTIFIC = re.compile(r"(?<![\w.])(?:\d+\.\d+|\d+)[eE][+-]?\d+")
_MUL_DOT = re.compile(r"([\w)])\s*\.\s*([\w(])")
_CALL = re.compile(rf"({_IDENT})\s*\(")
_REDUNDANT_FRACTION = re.compile(r"1 \\frac\{1\}")

DISTRIBUTIONS = ("Normaal", "Rechthoekig", "Driehoekig")


class FormulaError(ValueError):
    """The measurement model or a quantity expression could not be evaluated."""


@dataclass
class ParsedModel:
    output: str
    expression: sympy.Expr
    quantities: list[str]
    latex: str


@dataclass
class TypeASummary:
    count: int
    mean: float
    stdev: float | None
    standard_uncertainty: float
    degrees_of_freedom: float
    pooled: bool = False
    direct: bool = False


@dataclass
class QuantityInput:
    name: str
    value_expression: str
    unit: str = ""
    kind: str = "B"
    distribution: str = "Normaal"
    uncertainty_expression: str = ""
    coverage_factor: float = 2.0
    degrees_of_freedom: float = math.inf
    comment: str = ""
    measurements: list[float] | None = None
    uncertainty_of_mean: bool = True
    use_mean_as_value: bool = True
    pooled_stdev: float | None = None
    pooled_degrees_of_freedom: float | None = None
    direct_stdev: float | None = None
    direct_count: int | None = None


@dataclass
class Contribution:
    name: str
    nominal: float
    unit: str
    kind: str
    distribution: str
    standard_uncertainty: float
    degrees_of_freedom: float
    sensitivity: float
    sensitivity_latex: str
    product: float
    share: float
    conversion: str
    comment: str


@dataclass
class Budget:
    output: str
    latex: str
    nominal: float
    combined: float
    degrees_of_freedom: float
    coverage_factor: float
    expanded: float
    confidence: float
    contributions: list[Contribution] = field(default_factory=list)

    @property
    def relative(self) -> float | None:
        if self.nominal == 0:
            return None
        return abs(self.combined / self.nominal)


@dataclass
class MonteCarloResult:
    output: str
    trials: int
    accepted: int
    seed: int
    mean: float
    standard_uncertainty: float
    low: float
    high: float
    confidence: float
    edges: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    heavy_tail: bool = False


def preprocess(expression: str) -> str:
    """Normalize a student-facing expression for SymPy."""
    text = expression.strip()
    text = (
        text.replace("π", "pi")
        .replace("√", "sqrt")
        .replace("·", "*")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )
    text = _DECIMAL_COMMA.sub(r"\1.\2", text)
    text = _PERCENT.sub(r"(\1/100)", text)
    text = re.sub(r"\bln\b", "log", text)
    return _explicit_products(text)


def _explicit_products(text: str) -> str:
    """Turn a multiplication dot, and a number or bracket beside a quantity, into *."""
    protected: list[str] = []

    def hold(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"@@SCI{len(protected) - 1}@@"

    text = _SCIENTIFIC.sub(hold, text)

    def dot(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if left.isdigit() and right.isdigit():
            return match.group(0)
        return f"{left}*{right}"

    text = _MUL_DOT.sub(dot, text)
    text = re.sub(r"\)\s*\(", ")*(", text)
    text = re.sub(rf"\)\s*({_IDENT})", r")*\1", text)
    text = re.sub(r"(\d)\s*\(", r"\1*(", text)
    text = re.sub(rf"(\d)\s*({_IDENT})", r"\1*\2", text)

    def call(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in _LOCALS and name not in _CONSTANTS:
            return match.group(0)
        return f"{name}*("

    text = _CALL.sub(call, text)
    for index, token in enumerate(protected):
        text = text.replace(f"@@SCI{index}@@", token)
    return text


def parse_expression(expression: str) -> sympy.Expr:
    prepared = preprocess(expression)
    if not prepared:
        raise FormulaError("lege formule")
    try:
        return parse_expr(
            prepared,
            local_dict=_LOCALS,
            global_dict=_GLOBAL,
            transformations=_TRANSFORMS,
            evaluate=False,
        )
    except Exception as exc:
        raise FormulaError(f"ongeldige formule: {exc}") from exc


def _presentation_latex(expression: sympy.Expr) -> str:
    """Latex that keeps the entered order and hides a duplicated numerator 1."""
    return _REDUNDANT_FRACTION.sub(r"\\frac{1}", sympy.latex(expression, order="none"))


def parse_model(formula: str) -> ParsedModel:
    text = formula.strip()
    if not text:
        raise FormulaError("voer een meetmodel in")
    if text.count("=") > 1:
        raise FormulaError("gebruik hooguit één =-teken")

    if "=" in text:
        raw_name, raw_rhs = text.split("=", 1)
        output = raw_name.strip()
        if not _NAME.match(output):
            raise FormulaError("de naam links van = is ongeldig")
        rhs_source = raw_rhs
    else:
        output = "y"
        rhs_source = text

    expression = parse_expression(rhs_source)
    if not isinstance(expression, sympy.Expr):
        raise FormulaError("het rechterlid is geen wiskundige uitdrukking")

    output_symbol = sympy.Symbol(output)
    if output_symbol in expression.free_symbols or output in {str(s) for s in expression.free_symbols}:
        raise FormulaError("de uitkomst mag niet ook in het rechterlid staan")

    names = _ordered_quantities(rhs_source, expression)
    equation = sympy.Eq(output_symbol, expression, evaluate=False)
    return ParsedModel(
        output=output,
        expression=expression,
        quantities=names,
        latex=_presentation_latex(equation),
    )


def _ordered_quantities(source: str, expression: sympy.Expr) -> list[str]:
    prepared = preprocess(source)
    names = [str(symbol) for symbol in expression.free_symbols]

    def position(name: str) -> int:
        match = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", prepared)
        return match.start() if match else 10**6

    return sorted(names, key=position)


def prepare_uncertainty_expression(expression: str, quantity: str) -> str:
    """Turn a bare percentage into a fraction of this quantity's nominal value."""
    original = expression.strip()
    if not original:
        return ""
    prepared = preprocess(original)
    if "%" not in original:
        return prepared
    parsed = parse_expression(original)
    if parsed.free_symbols:
        return prepared
    return f"({prepared})*({quantity})"


_SERIES_SPLIT = re.compile(r"[\s;]+")
_DECIMAL_COMMA_NUMBER = re.compile(r"^[+-]?\d+,\d+$")
_PLAIN_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_COMMA_LIST = re.compile(r"^[+-]?\d+(?:,[+-]?\d+)+$")


def parse_number_series(text: str) -> list[float]:
    """Read a typed series. Decimal commas, spaces, newlines and semicolons are allowed."""
    raw = text.strip()
    if not raw:
        return []
    values: list[float] = []
    for chunk in _SERIES_SPLIT.split(raw):
        if not chunk:
            continue
        if _COMMA_LIST.fullmatch(chunk) and not _DECIMAL_COMMA_NUMBER.fullmatch(chunk):
            values.extend(float(part) for part in chunk.split(","))
            continue
        if _DECIMAL_COMMA_NUMBER.fullmatch(chunk):
            values.append(float(chunk.replace(",", ".")))
            continue
        if _PLAIN_NUMBER.fullmatch(chunk):
            values.append(float(chunk))
            continue
        raise FormulaError(f"ongeldig getal in de meetreeks: {chunk}")
    if len(values) > 100_000:
        raise FormulaError("te veel meetwaarden (maximum 100 000)")
    return values


def summarize_type_a(
    values: list[float] | np.ndarray,
    uncertainty_of_mean: bool = True,
    pooled_stdev: float | None = None,
    pooled_degrees_of_freedom: float | None = None,
) -> TypeASummary:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    count = int(data.size)
    if count < 1:
        raise FormulaError("voer minstens één meetwaarde in")
    if count > 100_000:
        raise FormulaError("te veel meetwaarden (maximum 100 000)")
    mean = float(np.mean(data))
    stdev = float(np.std(data, ddof=1)) if count >= 2 else None
    if pooled_stdev is not None:
        if pooled_stdev < 0 or not math.isfinite(pooled_stdev):
            raise FormulaError("de pooled estimate moet 0 of groter zijn")
        if (
            pooled_degrees_of_freedom is None
            or not math.isfinite(pooled_degrees_of_freedom)
            or pooled_degrees_of_freedom <= 0
        ):
            raise FormulaError("vul de vrijheidsgraden van de pooled estimate in")
        return TypeASummary(
            count=count,
            mean=mean,
            stdev=stdev,
            standard_uncertainty=pooled_stdev / math.sqrt(count),
            degrees_of_freedom=float(pooled_degrees_of_freedom),
            pooled=True,
        )
    if count < 2 or stdev is None:
        raise FormulaError("een Type A-reeks heeft minstens twee meetwaarden nodig, of een pooled estimate")
    standard = stdev / math.sqrt(count) if uncertainty_of_mean else stdev
    return TypeASummary(
        count=count,
        mean=mean,
        stdev=stdev,
        standard_uncertainty=standard,
        degrees_of_freedom=float(count - 1),
    )


def summarize_direct_stdev(stdev: float, count: int) -> TypeASummary:
    """Standard uncertainty from an entered standard deviation and observation count."""
    if not math.isfinite(stdev) or stdev < 0:
        raise FormulaError("de standaardafwijking moet 0 of groter zijn")
    if count < 1:
        raise FormulaError("het aantal waarnemingen moet minstens 1 zijn")
    return TypeASummary(
        count=count,
        mean=0.0,
        stdev=float(stdev),
        standard_uncertainty=float(stdev) / math.sqrt(count),
        degrees_of_freedom=float(count - 1) if count >= 2 else math.inf,
        direct=True,
    )


def read_measurement_table(data: bytes) -> pd.DataFrame:
    """Read a CSV of measurement values, including Dutch decimal commas."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FormulaError("het bestand is geen tekstbestand") from exc
    if not text.strip():
        raise FormulaError("het CSV-bestand is leeg")

    best: pd.DataFrame | None = None
    best_score = (-1, -1)
    attempts = (
        {"sep": ";", "decimal": ","},
        {"sep": ",", "decimal": "."},
        {"sep": "\t", "decimal": "."},
        {"sep": "\t", "decimal": ","},
        {"sep": ";", "decimal": "."},
    )
    for header in (0, None):
        for options in attempts:
            try:
                frame = pd.read_csv(io.StringIO(text), header=header, **options)
            except Exception:
                continue
            score = (_numeric_cells(frame), int(frame.shape[1]))
            if score > best_score:
                best = frame
                best_score = score
    if best is None or best_score[0] <= 0:
        raise FormulaError("geen meetwaarden in het CSV-bestand gevonden")
    best.columns = [str(column).strip() for column in best.columns]
    return best


def numeric_column(frame: pd.DataFrame, column: str) -> list[float]:
    if column not in frame.columns:
        raise FormulaError(f"kolom {column} ontbreekt")
    series = frame[column]
    if series.dtype.kind not in "iuf":
        series = pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise FormulaError(f"kolom {column} bevat geen getallen")
    return [float(value) for value in values]


def richest_column(frame: pd.DataFrame) -> str:
    scored = [( _series_numeric_count(frame[column]), str(column)) for column in frame.columns]
    scored.sort(reverse=True)
    return scored[0][1]


def evaluate(model: ParsedModel, quantities: list[QuantityInput], confidence: float = 0.95) -> Budget:
    if not 0 < confidence < 1:
        raise FormulaError("het betrouwbaarheidsniveau moet tussen 0 en 1 liggen")
    by_name = {item.name: item for item in quantities}
    missing = [name for name in model.quantities if name not in by_name]
    if missing:
        raise FormulaError("ontbrekende grootheden: " + ", ".join(missing))

    nominals = _resolve_nominals(model.quantities, by_name)
    contributions: list[Contribution] = []
    for name in model.quantities:
        item = by_name[name]
        sensitivity, sensitivity_latex = _sensitivity(model.expression, name, nominals)
        standard, dof, distribution, conversion = _standard_uncertainty(item, nominals)
        product = sensitivity * standard
        contributions.append(
            Contribution(
                name=name,
                nominal=nominals[name],
                unit=item.unit,
                kind=item.kind,
                distribution=distribution,
                standard_uncertainty=standard,
                degrees_of_freedom=dof,
                sensitivity=sensitivity,
                sensitivity_latex=sensitivity_latex,
                product=product,
                share=0.0,
                conversion=conversion,
                comment=item.comment,
            )
        )

    combined = math.sqrt(sum(item.product**2 for item in contributions))
    nu_eff = _welch_satterthwaite(combined, contributions)
    k = _coverage_factor(confidence, nu_eff)
    if combined > 0:
        for item in contributions:
            item.share = 100.0 * (item.product**2) / combined**2

    output_nominal = _as_float(model.expression.subs({sympy.Symbol(name): value for name, value in nominals.items()}))
    return Budget(
        output=model.output,
        latex=model.latex,
        nominal=output_nominal,
        combined=combined,
        degrees_of_freedom=nu_eff,
        coverage_factor=k,
        expanded=k * combined,
        confidence=confidence,
        contributions=contributions,
    )


@dataclass
class SweepCurve:
    variable: str
    values: list[float]
    nominals: list[float]
    expanded: list[float]
    skipped: int


def logical_step(span: float) -> float:
    """Pick a round step between 5% and 15% of the span, nearest to 10%."""
    if not math.isfinite(span) or span <= 0:
        raise FormulaError("het bereik moet een positieve lengte hebben")
    low, high, target = 0.05 * span, 0.15 * span, 0.10 * span
    exponent = math.floor(math.log10(span))
    best: float | None = None
    best_distance = math.inf
    for power in range(exponent - 4, exponent + 2):
        for mantissa in (1.0, 2.0, 2.5, 5.0):
            step = mantissa * 10.0**power
            if step < low * (1 - 1e-9) or step > high * (1 + 1e-9):
                continue
            distance = abs(step - target)
            if distance < best_distance - 1e-12:
                best, best_distance = step, distance
    if best is None:
        raise FormulaError("er is geen passende stapgrootte voor dit bereik")
    return best


def range_points(start: float, stop: float) -> tuple[float, list[float]]:
    """Endpoints plus the round values in between, spaced by logical_step."""
    step = logical_step(abs(stop - start))
    lo, hi = (start, stop) if start <= stop else (stop, start)
    first = math.ceil((lo - step * 1e-9) / step) * step
    last = math.floor((hi + step * 1e-9) / step) * step
    points = [lo]
    count = int(round((last - first) / step))
    for index in range(count + 1):
        value = float(f"{(first + index * step):.12g}")
        if value <= lo + step * 1e-9 or value >= hi - step * 1e-9:
            continue
        points.append(value)
    if hi > lo + step * 1e-9:
        points.append(hi)
    if start > stop:
        points.reverse()
    return step, points


def sweep(
    model: ParsedModel,
    quantities: list[QuantityInput],
    variable: str,
    start: float,
    stop: float,
    confidence: float = 0.95,
    count: int = 101,
    points: list[float] | None = None,
) -> SweepCurve:
    """Recompute the model and its expanded uncertainty along one input."""
    if variable not in {item.name for item in quantities}:
        raise FormulaError(f"{variable} hoort niet bij dit model")
    if points is None and count < 2:
        raise FormulaError("het bereik heeft minstens twee punten nodig")
    if not math.isfinite(start) or not math.isfinite(stop):
        raise FormulaError("de grenzen van het bereik moeten eindige getallen zijn")
    if start == stop:
        raise FormulaError("de grenzen van het bereik moeten verschillend zijn")

    sequence = list(np.linspace(start, stop, count)) if points is None else points
    values: list[float] = []
    nominals: list[float] = []
    expanded: list[float] = []
    skipped = 0
    for point in sequence:
        varied = [
            replace(item, value_expression=repr(float(point)), use_mean_as_value=False)
            if item.name == variable
            else item
            for item in quantities
        ]
        try:
            budget = evaluate(model, varied, confidence)
        except FormulaError:
            skipped += 1
            continue
        values.append(float(point))
        nominals.append(budget.nominal)
        expanded.append(budget.expanded)
    if not values:
        raise FormulaError("het resultaat is op geen enkel punt in het bereik te berekenen")
    return SweepCurve(
        variable=variable,
        values=values,
        nominals=nominals,
        expanded=expanded,
        skipped=skipped,
    )


def _resolve_nominals(names: list[str], by_name: dict[str, QuantityInput]) -> dict[str, float]:
    pending = {name: by_name[name] for name in names}
    resolved: dict[str, float] = {}
    for _ in range(len(names) + 1):
        if not pending:
            return resolved
        progress = False
        for name, item in list(pending.items()):
            expression = _nominal_expression(item)
            symbols = {str(symbol) for symbol in parse_expression(expression).free_symbols}
            if name in symbols:
                raise FormulaError(f"de waarde van {name} mag niet van zichzelf afhangen")
            unknown = symbols - set(names)
            if unknown:
                raise FormulaError(
                    f"de waarde van {name} gebruikt onbekende grootheden: " + ", ".join(sorted(unknown))
                )
            if any(symbol in pending for symbol in symbols):
                continue
            resolved[name] = _eval_number(expression, resolved, f"de waarde van {name}")
            del pending[name]
            progress = True
        if not progress:
            break
    joined = ", ".join(sorted(pending))
    raise FormulaError(f"de waarden van {joined} verwijzen kringvormig naar elkaar")


def _type_a_summary(item: QuantityInput) -> TypeASummary:
    if item.direct_stdev is not None and item.direct_count is not None:
        try:
            return summarize_direct_stdev(item.direct_stdev, item.direct_count)
        except FormulaError as exc:
            raise FormulaError(f"{exc} ({item.name})") from exc
    if not item.measurements:
        raise FormulaError(f"voer meetwaarden in voor {item.name}")
    try:
        return summarize_type_a(
            item.measurements,
            item.uncertainty_of_mean,
            item.pooled_stdev,
            item.pooled_degrees_of_freedom,
        )
    except FormulaError as exc:
        message = str(exc)
        if message.startswith("vul de") or message.startswith("de pooled"):
            raise FormulaError(f"{message} ({item.name})") from exc
        raise


def _nominal_expression(item: QuantityInput) -> str:
    if item.kind == "A" and item.use_mean_as_value:
        summary = _type_a_summary(item)
        return repr(summary.mean)
    text = item.value_expression.strip()
    if not text:
        raise FormulaError(f"vul de waarde van {item.name} in")
    return text


def _standard_uncertainty(
    item: QuantityInput, nominals: dict[str, float]
) -> tuple[float, float, str, str]:
    if item.kind == "A":
        summary = _type_a_summary(item)
        if summary.direct and summary.stdev is not None:
            conversion = f"s/√n = {_plain(summary.stdev)}/√{summary.count}"
        elif summary.pooled and item.pooled_stdev is not None:
            conversion = f"s_p/√n = {_plain(item.pooled_stdev)}/√{summary.count}"
        elif item.uncertainty_of_mean and summary.stdev is not None:
            conversion = f"s/√n = {_plain(summary.stdev)}/√{summary.count}"
        elif summary.stdev is not None:
            conversion = f"s = {_plain(summary.stdev)}"
        else:
            conversion = "s_p/√n"
        return summary.standard_uncertainty, float(summary.degrees_of_freedom), "Type A (t)", conversion

    if item.distribution not in DISTRIBUTIONS:
        raise FormulaError(f"onbekende verdeling voor {item.name}")
    raw_text = item.uncertainty_expression.strip()
    if not raw_text:
        raise FormulaError(f"vul de onzekerheid van {item.name} in")
    prepared = prepare_uncertainty_expression(raw_text, item.name)
    raw = _eval_number(prepared, nominals, f"de onzekerheid van {item.name}")
    if raw < 0:
        raise FormulaError(f"de onzekerheid van {item.name} mag niet negatief zijn")

    if item.distribution == "Normaal":
        if item.coverage_factor <= 0:
            raise FormulaError(f"k van {item.name} moet positief zijn")
        standard = raw / item.coverage_factor
        conversion = f"{_plain(raw)} / { _plain(item.coverage_factor)}"
        return standard, item.degrees_of_freedom, "Normaal", conversion
    if item.distribution == "Rechthoekig":
        return raw / math.sqrt(3), item.degrees_of_freedom, "Rechthoekig", f"{_plain(raw)} / √3"
    return raw / math.sqrt(6), item.degrees_of_freedom, "Driehoekig", f"{_plain(raw)} / √6"


def _sensitivity(expression: sympy.Expr, name: str, nominals: dict[str, float]) -> tuple[float, str]:
    derivative = sympy.diff(expression, sympy.Symbol(name))
    value = _as_float(derivative.subs({sympy.Symbol(key): nominal for key, nominal in nominals.items()}))
    return value, sympy.latex(derivative)


def _eval_number(expression: str, known: dict[str, float], label: str) -> float:
    parsed = parse_expression(expression)
    substituted = parsed.subs({sympy.Symbol(key): value for key, value in known.items()})
    remaining = sorted(str(symbol) for symbol in substituted.free_symbols)
    if remaining:
        raise FormulaError(f"{label} gebruikt onbekende grootheden: " + ", ".join(remaining))
    try:
        return _as_float(substituted)
    except FormulaError as exc:
        raise FormulaError(f"{label} is geen getal") from exc


def _as_float(value) -> float:
    try:
        number = complex(sympy.N(value))
    except Exception as exc:
        raise FormulaError("het resultaat is niet te berekenen") from exc
    if not math.isfinite(number.real) or abs(number.imag) > 1e-9:
        raise FormulaError("het resultaat is niet een eindig reëel getal")
    return float(number.real)


def _welch_satterthwaite(combined: float, contributions: list[Contribution]) -> float:
    if combined == 0:
        return math.inf
    denominator = 0.0
    for item in contributions:
        if item.product == 0 or math.isinf(item.degrees_of_freedom):
            continue
        if item.degrees_of_freedom <= 0:
            raise FormulaError(f"vrijheidsgraden van {item.name} moeten positief zijn")
        denominator += item.product**4 / item.degrees_of_freedom
    if denominator == 0:
        return math.inf
    return combined**4 / denominator


def _coverage_factor(confidence: float, degrees_of_freedom: float) -> float:
    probability = (1 + confidence) / 2
    if math.isinf(degrees_of_freedom):
        return float(stats.norm.ppf(probability))
    return float(stats.t.ppf(probability, degrees_of_freedom))


def monte_carlo(
    model: ParsedModel,
    quantities: list[QuantityInput],
    confidence: float = 0.95,
    trials: int = 100_000,
    seed: int = 1,
) -> MonteCarloResult:
    """Propagate independent draws through the model and summarize the output."""
    if not 0 < confidence < 1:
        raise FormulaError("het betrouwbaarheidsniveau moet tussen 0 en 1 liggen")
    if trials < 1_000 or trials > 1_000_000:
        raise FormulaError("het aantal trekkingen moet tussen 1 000 en 1 000 000 liggen")
    if seed < 0 or seed > 2**32 - 1:
        raise FormulaError("de startwaarde moet tussen 0 en 4294967295 liggen")

    by_name = {item.name: item for item in quantities}
    missing = [name for name in model.quantities if name not in by_name]
    if missing:
        raise FormulaError("ontbrekende grootheden: " + ", ".join(missing))

    nominals = _resolve_nominals(model.quantities, by_name)
    rng = np.random.default_rng(int(seed))
    columns: list[np.ndarray] = []
    heavy_tail = False
    for name in model.quantities:
        item = by_name[name]
        standard, dof, _, _ = _standard_uncertainty(item, nominals)
        if (item.kind == "A" or item.distribution == "Normaal") and math.isfinite(dof) and dof <= 2:
            heavy_tail = True
        columns.append(_draw_quantity(rng, item, nominals[name], standard, dof, trials))

    values = _evaluate_draws(model, columns, trials)
    accepted = values[np.isfinite(values)]
    if accepted.size < 2:
        raise FormulaError("te weinig geldige trekkingen om een onzekerheid te bepalen")

    low_probability = (1 - confidence) / 2
    low, high = np.quantile(accepted, [low_probability, 1 - low_probability])
    edges, counts = _output_histogram(accepted)
    return MonteCarloResult(
        output=model.output,
        trials=trials,
        accepted=int(accepted.size),
        seed=int(seed),
        mean=float(np.mean(accepted)),
        standard_uncertainty=float(np.std(accepted, ddof=1)),
        low=float(low),
        high=float(high),
        confidence=confidence,
        edges=edges,
        counts=counts,
        heavy_tail=heavy_tail,
    )


def _draw_quantity(
    rng: np.random.Generator,
    item: QuantityInput,
    center: float,
    standard: float,
    degrees_of_freedom: float,
    trials: int,
) -> np.ndarray:
    """Draw one input from the distribution already used in the GUM budget."""
    if item.kind == "A" or item.distribution == "Normaal":
        if standard == 0:
            return np.full(trials, center)
        if math.isfinite(degrees_of_freedom) and degrees_of_freedom < 1_000_000:
            return center + standard * rng.standard_t(degrees_of_freedom, size=trials)
        return rng.normal(center, standard, size=trials)

    half_width = standard * math.sqrt(3 if item.distribution == "Rechthoekig" else 6)
    if half_width == 0:
        return np.full(trials, center)
    if item.distribution == "Rechthoekig":
        return rng.uniform(center - half_width, center + half_width, size=trials)
    return rng.triangular(center - half_width, center, center + half_width, size=trials)


def _evaluate_draws(model: ParsedModel, columns: list[np.ndarray], trials: int) -> np.ndarray:
    if columns:
        symbols = [sympy.Symbol(name) for name in model.quantities]
        function = sympy.lambdify(symbols, model.expression, modules=["numpy"])
        output = function(*columns)
    else:
        output = _as_float(model.expression)
    values = np.asarray(output)
    if np.iscomplexobj(values):
        imag = np.abs(np.imag(values))
        values = np.where(imag <= 1e-9 * np.maximum(1.0, np.abs(np.real(values))), np.real(values), np.nan)
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1 and trials > 1:
        values = np.full(trials, float(values[0]))
    if values.size != trials:
        raise FormulaError("de formule geeft niet één uitkomst per trekking")
    return values


def _output_histogram(values: np.ndarray, bins: int = 40) -> tuple[list[float], list[int]]:
    if float(np.max(values) - np.min(values)) == 0:
        return [], []
    counts, edges = np.histogram(values, bins=bins)
    return [float(edge) for edge in edges], [int(count) for count in counts]


def _numeric_cells(frame: pd.DataFrame) -> int:
    return sum(_series_numeric_count(frame[column]) for column in frame.columns)


def _series_numeric_count(series: pd.Series) -> int:
    if series.dtype.kind in "iuf":
        return int(np.isfinite(series.to_numpy(dtype=float)).sum())
    converted = pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return int(converted.notna().sum())


def _plain(value: float) -> str:
    return f"{value:.6g}".replace(".", ",")
