from types import SimpleNamespace

import pytest

from tianximem.common.tokens import O200kCounter
from tianximem.facts.input import input_complete
from tianximem.retrieve import EvidenceChecker
from tianximem.service.pipeline import SearchPipeline
from tianximem.store.sqlite_store import SqliteStore

CSV_ROW = (
    "Convert the raw values below into CSV with exactly this column order: name,enabled\n"
    "name=example\nenabled=true\nReturn only the header and one row."
)
CSV_MAP = (
    "Take the raw input data below and output plain CSV "
    "with these columns: field_name,field_value. "
    "No explanation.\nRaw input data:\n- account_id = 17\n- enabled = true"
)
FIELD = "What is the 'target_host' host string in this config snippet?\ntarget_host: example.test"
PACKAGE = (
    'Here is the package.json fragment:\n{"dependencies":{"example-lib":"2.3.4"}}\n'
    "What is the version number of the 'example-lib' library in this package.json?"
)
METRICS = (
    'Raw metrics scrap:\nup{job="node",instance="example"} 1\n'
    'requests{Job="payments"} 12\ncustom{job_name="checkout"} 3\n\n'
    "Just list the Job names found in a simple bulleted list."
)


@pytest.mark.parametrize(
    "query",
    [
        CSV_ROW,
        CSV_MAP,
        CSV_MAP.replace("account_id = 17", "account_id = $12.50"),
        FIELD,
        FIELD + "\ntarget_host: example.test",
        PACKAGE,
        METRICS,
    ],
)
def test_complete_current_literal_payload_does_not_need_history(query):
    assert input_complete(query)


@pytest.mark.parametrize(
    "query",
    [
        "Return the CSV using our stored values.",
        "What is the 'target_host' host string in this config file?",
        FIELD + "\ntarget_host: other.test",
        FIELD.replace("example.test", "${HOST}"),
        FIELD.replace("example.test", "$HOST"),
        FIELD.replace("example.test", "/service/$HOST"),
        FIELD.replace("example.test", "unknown"),
        FIELD.replace("target_host:", "Target_Host:"),
        "Compare this config to the previous one. " + FIELD,
        "Apply the policy we agreed on. " + CSV_MAP,
        CSV_ROW.replace("name,enabled", "name,enabled,region"),
        CSV_ROW.replace("raw values", "values"),
        CSV_MAP.replace("account_id = 17\n", ""),
        CSV_MAP.replace("account_id = 17", "account_id = $VALUE"),
        "What is the total amount I earned from selling at the markets?",
        "What were the names of my pets?",
        "Given a 50 by 50 matrix, list all eigenvalues.",
        "Output these public input values as base-10: pub_a=002a, pub_b=0007.",
        PACKAGE.replace("example-lib", "missing-lib", 1),
        PACKAGE.replace('"2.3.4"', '"^2.3.4"'),
        PACKAGE.replace('"2.3.4"', "null"),
        PACKAGE.replace('"2.3.4"', '"${VERSION}"'),
        PACKAGE.replace('"example-lib":"2.3.4"', '"example-lib":"2.3.4","example-lib":"2.3.5"'),
        PACKAGE.replace("}}", '},"optionalDependencies":{"example-lib":"2.3.5"}}'),
        PACKAGE.replace("}}", '},"dependencies":{"example-lib":"2.3.5"}}'),
        PACKAGE.replace("}}\n", '}}\n{"dependencies":{"example-lib":"2.3.5"}}\n'),
        PACKAGE.replace("}}", "}"),
        "Use our existing policy. " + PACKAGE,
        "Use our usual format. " + METRICS,
        METRICS.replace('job="node"', 'job="${JOB}"'),
        METRICS.replace('job="node"', 'job="$JOB"'),
        METRICS.replace('job="node"', 'job=""'),
        METRICS.replace('job="node"', 'job="node",job="other"'),
        METRICS.replace('Job="payments"', 'Job="payments'),
        METRICS.replace('requests{Job="payments"} 12', "invalid metric line"),
        METRICS.replace('requests{Job="payments"} 12', '\nrequests{Job="payments"} 12'),
        METRICS.replace('job="node"', 'zone="node"')
        .replace('Job="payments"', 'zone="payments"')
        .replace('job_name="checkout"', 'zone="checkout"'),
    ],
)
def test_missing_conflicting_or_historical_inputs_keep_original_retrieval(query):
    assert not input_complete(query)


def test_search_only_skips_retrieval_when_enabled_and_payload_is_complete(tmp_path):
    store = SqliteStore.open(tmp_path / "test.db")
    exists_calls = []

    def exists():
        exists_calls.append(True)
        return False

    def pipeline(enabled):
        return SearchPipeline(
            store=store,
            qdrant=SimpleNamespace(exists=exists),
            retriever=None,
            checker=EvidenceChecker(),
            counter=O200kCounter(),
            budget_tokens=117760,
            grounded_evidence=enabled,
        )

    assert not pipeline(True).run(user_id="u", query=CSV_MAP, top_k=100).items
    assert not exists_calls
    assert not pipeline(False).run(user_id="u", query=CSV_MAP, top_k=100).items
    assert len(exists_calls) == 1
    assert not pipeline(True).run(user_id="u", query="Return our stored CSV.", top_k=100).items
    assert len(exists_calls) == 2
    assert (
        not pipeline(True)
        .run(user_id="u", query=METRICS.replace('job="node"', 'job="$JOB"'), top_k=100)
        .items
    )
    assert len(exists_calls) == 3
