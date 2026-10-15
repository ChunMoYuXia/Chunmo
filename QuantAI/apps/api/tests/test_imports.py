import pytest


def test_corrected_parser_does_not_reuse_a_legacy_dataset_content_identity():
    import hashlib

    from portfolio_client.infrastructure.imports.csv import preview_import

    payload = b"code,date,open,high,low,close,volume\n920005,2026-01-05,10,11,9,10,1000\n"
    legacy_hash = hashlib.sha256(payload + b"{}shares").hexdigest()
    preview = preview_import(payload, mapping={})
    assert preview.content_hash != legacy_hash
    assert preview.parser_version == "csv-v2"


@pytest.mark.parametrize("code", ["920005", "920005.BJ"])
def test_beijing_920_codes_keep_exchange_identity_and_reject_execution(code):
    from portfolio_client.domain.errors import DomainError
    from portfolio_client.domain.execution.rules import lot_size_for
    from portfolio_client.infrastructure.imports.csv import preview_import

    payload = f"code,date,open,high,low,close,volume\n{code},2026-01-05,10,11,9,10,1000\n"
    preview = preview_import(payload.encode(), mapping={})
    assert not preview.errors
    security = preview.rows[0].security
    assert security.security_id == "XBSE:920005" and security.board == "bse"
    with pytest.raises(DomainError) as error:
        lot_size_for(security)
    assert error.value.code == "UNSUPPORTED_MARKET_RULES"


@pytest.mark.parametrize("code,exchange", [("920005.SH", ""), ("920005", "XSHE")])
def test_beijing_920_rejects_conflicting_explicit_exchange(code, exchange):
    from portfolio_client.infrastructure.imports.csv import preview_import

    payload = f"code,exchange,date,open,high,low,close,volume\n{code},{exchange},2026-01-05,10,11,9,10,1000\n"
    preview = preview_import(payload.encode(), mapping={})
    assert not preview.rows
    assert preview.errors[0].code == "INVALID_EXCHANGE" and preview.errors[0].row == 2


def test_shanghai_900_codes_keep_their_exchange_identity():
    from portfolio_client.infrastructure.imports.csv import preview_import

    preview = preview_import(
        b"code,date,open,high,low,close,volume\n900901,2026-01-05,10,11,9,10,1000\n", mapping={}
    )
    assert not preview.errors
    assert preview.rows[0].security.security_id == "XSHG:900901"


def test_chinese_csv_preserves_leading_zeros_and_reports_row_number() -> None:
    from portfolio_client.infrastructure.imports.csv import preview_import

    payload = (
        "\ufeff代码,日期,开盘,最高,最低,收盘,成交量\n"
        "000001,2026-01-05,10,11,9,10.5,1000\n"
        "000002,2026-01-05,12,10,11,12,1000\n"
    ).encode()
    preview = preview_import(payload, mapping={})
    assert preview.rows[0].code == "000001"
    assert preview.rows[0].bar.security_id == "XSHE:000001"
    assert preview.errors[0].row == 3
    assert preview.errors[0].code == "INVALID_OHLC"


@pytest.mark.parametrize("price", ["NaN", "Infinity", "0", "-1", "1e1000"])
def test_nonfinite_or_nonpositive_price_is_rejected(price: str) -> None:
    from portfolio_client.infrastructure.imports.csv import preview_import

    content = f"code,date,open,high,low,close,volume\n000001,2026-01-05,{price},11,9,10,1000\n"
    preview = preview_import(content.encode(), mapping={})
    assert preview.errors[0].code == "INVALID_PRICE"
    assert not preview.rows


def test_duplicate_security_session_is_a_visible_error() -> None:
    from portfolio_client.infrastructure.imports.csv import preview_import

    content = "code,date,open,high,low,close,volume\n000001,2026-01-05,10,11,9,10,1000\n000001,2026-01-05,10,11,9,10,1000\n"
    preview = preview_import(content.encode(), mapping={})
    assert preview.errors[0].code == "DUPLICATE_BAR"


def test_synthetic_seed_is_reproducible_and_data_is_identified() -> None:
    from portfolio_client.infrastructure.imports.synthetic import generate_dataset

    first = generate_dataset(seed=7, securities=3, sessions=130)
    second = generate_dataset(seed=7, securities=3, sessions=130)
    assert first.content_hash == second.content_hash
    assert first.rows == second.rows
    assert first.source_kind == "synthetic"
    assert len(first.rows) == 390
    assert first.actions


def test_volume_must_fit_persistent_integer_range() -> None:
    from portfolio_client.infrastructure.imports.csv import preview_import

    content = f"code,date,open,high,low,close,volume\n000001,2026-01-05,10,11,9,10,{2**80}\n"
    preview = preview_import(content.encode(), mapping={})
    assert preview.errors[0].code == "INVALID_VOLUME"


def test_calendar_rejects_unknown_open_flag() -> None:
    from portfolio_client.domain.errors import DomainError
    from portfolio_client.infrastructure.imports.csv import parse_calendar

    with pytest.raises(DomainError) as error:
        parse_calendar(b"date,is_open\n2026-01-05,unknown\n")
    assert error.value.code == "INVALID_CALENDAR"


def test_duplicate_headers_are_not_silently_overwritten() -> None:
    from portfolio_client.infrastructure.imports.csv import preview_import

    content = (
        b"code,code,date,open,high,low,close,volume\n000001,000002,2026-01-05,10,11,9,10,1000\n"
    )
    preview = preview_import(content, mapping={})
    assert preview.errors[0].code == "DUPLICATE_COLUMNS"
