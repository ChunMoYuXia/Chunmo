import re

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.research.metric_models import Metric
from portfolio_client.domain.research.reports import format_metric

TOKEN = re.compile(r"\{\{metric:([a-zA-Z0-9_]+)\}\}")


def render_metric_tokens(text: str, metrics: dict[str, Metric]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in metrics:
            raise DomainError("UNKNOWN_METRIC_REFERENCE", "指标引用不存在")
        return format_metric(metrics[key])

    rendered = TOKEN.sub(replace, text)
    remaining = TOKEN.sub("", text)
    if "{{" in remaining or "}}" in remaining:
        raise DomainError("INVALID_METRIC_REFERENCE", "指标引用格式无效")
    # Deliberately conservative: Chinese numerals are also forbidden outside metric tokens.
    if any(char.isnumeric() for char in remaining) or re.search(
        r"半|倍|翻番|腰斩|[几数]成", remaining
    ):
        raise DomainError("UNSOURCED_NUMBER", "解释中的数值必须引用后端指标")
    return rendered
