"""RPC transport behaviour: failover, batch splitting and error classification.

These are the paths that decide whether a result is real or silently missing,
which is why they get their own tests rather than being trusted to integration
runs against endpoints whose behaviour changes without notice.
"""
from __future__ import annotations

import time

import pytest

from dex_data.config import Config
from dex_data.rpc import RpcClient, RpcError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DEX_RPC_URLS", "https://a.example,https://b.example")
    monkeypatch.setenv("DEX_RPC_RATE_LIMIT", "0")
    return RpcClient(Config())


class TestErrorClassification:
    @pytest.mark.parametrize("message", [
        "Batch of more than 3 requests are not allowed on free plan",
        "too many RPC calls in batch request",
        "batch too large",
        "exceeds the limit of batch size",
    ])
    def test_batch_limits_are_recognised(self, message):
        assert RpcError(message).is_batch_too_large

    @pytest.mark.parametrize("message", [
        "Unauthorized: You must authenticate your request with an API key",
        "exceeded quota",
        "payment required",
    ])
    def test_endpoint_faults_are_recognised(self, message):
        assert RpcError(message).is_endpoint_fault

    @pytest.mark.parametrize("message", [
        "rpc method is not whitelisted",
        "Please specify an address in your request",
        "Archive requests require a personal token",
        "order a dedicated full node",
    ])
    def test_method_refusals_are_not_endpoint_faults(self, message):
        """These say 'not this method, here', not 'this endpoint is dead'.
        Blacklisting the method keeps the endpoint for the calls it does serve."""
        error = RpcError(message)
        assert error.is_method_unsupported
        assert not error.is_endpoint_fault

    @pytest.mark.parametrize("message", [
        "query returned more than 10000 results",
        "block range is too large",
        "Log response size exceeded",
    ])
    def test_range_limits_are_recognised(self, message):
        assert RpcError(message).is_range_too_large

    def test_range_limit_is_not_mistaken_for_a_dead_endpoint(self):
        """'limit exceeded' matches both vocabularies. It must mean 'ask for
        fewer blocks', never 'retire this endpoint'."""
        from dex_data.rpc import _endpoint_fault

        body = {"error": {"code": -32005, "message": "query returned more than 10000 results"}}
        assert _endpoint_fault(body) is None

    def test_execution_revert_is_neither(self):
        error = RpcError("execution reverted")
        assert not error.is_batch_too_large
        assert not error.is_endpoint_fault
        assert not error.is_range_too_large


class TestBatchSplitting:
    def test_identical_error_on_every_item_is_treated_as_batch_rejection(self, client):
        """The regression that motivated this test: drpc's free plan answers an
        oversized batch with a per-item error whose wording matches none of our
        patterns. Read literally that is 'every call reverted', so every result
        becomes None -- and a dropped factory() silently discards a real pool,
        which then breaks swap pairing far downstream. Identical errors across
        a whole batch must be retried, whatever the wording.
        """
        seen_sizes: list[int] = []

        def fake_post(payload, batch_size=1):
            seen_sizes.append(len(payload))
            endpoint = client._endpoints[0]
            if len(payload) > 3:
                return endpoint, [
                    {"jsonrpc": "2.0", "id": item["id"],
                     "error": {"code": -32029, "message": "Nonstandard wording nobody matches"}}
                    for item in payload
                ]
            return endpoint, [
                {"jsonrpc": "2.0", "id": item["id"], "result": "0x2a"} for item in payload
            ]

        client._post = fake_post
        results = client.batch([("eth_call", [{}, "latest"])] * 8)

        assert results == ["0x2a"] * 8, "no result may be silently dropped"
        assert max(seen_sizes) == 8 and min(seen_sizes) <= 3, "batch should have been split"

    def test_genuine_per_call_reverts_stay_none(self, client):
        """The opposite case: real reverts must NOT trigger endless retries.
        A v2 pool asked for fee() reverts, and None is the correct answer."""
        attempts = []

        def fake_post(payload, batch_size=1):
            attempts.append(len(payload))
            endpoint = client._endpoints[0]
            out = []
            for index, item in enumerate(payload):
                if index % 2:
                    out.append({"jsonrpc": "2.0", "id": item["id"],
                                "error": {"code": 3, "message": "execution reverted"}})
                else:
                    out.append({"jsonrpc": "2.0", "id": item["id"], "result": "0x01"})
            return endpoint, out

        client._post = fake_post
        results = client.batch([("eth_call", [{}, "latest"])] * 4)

        assert results == ["0x01", None, "0x01", None]
        assert len(attempts) == 1, "reverts must not cause a retry storm"

    def test_results_stay_aligned_with_calls_across_a_split(self, client):
        """Positional alignment is the whole contract of batch(); a shift would
        pair a pool with another pool's token addresses."""
        def fake_post(payload, batch_size=1):
            endpoint = client._endpoints[0]
            if len(payload) > 2:
                raise RpcError("too many RPC calls in batch request")
            return endpoint, [
                {"jsonrpc": "2.0", "id": item["id"],
                 "result": item["params"][0]["data"]}
                for item in payload
            ]

        client._post = fake_post
        calls = [("eth_call", [{"data": f"0x{i:02x}"}, "latest"]) for i in range(7)]
        assert client.batch(calls) == [f"0x{i:02x}" for i in range(7)]


class TestEndpointSelection:
    def test_learned_cap_steers_large_batches_away(self, client):
        small, large = client._endpoints
        small.max_batch = 3

        assert not small.accepts_batch(10)
        assert large.accepts_batch(10)
        assert small.accepts_batch(1), "single calls always go through"

        chosen = {client._next_endpoint(batch_size=10).url for _ in range(6)}
        assert chosen == {large.url}

    def test_cooldown_endpoint_is_skipped(self, client):
        dead, alive = client._endpoints
        dead.penalise(60)
        assert not dead.available
        chosen = {client._next_endpoint().url for _ in range(6)}
        assert chosen == {alive.url}

    def test_shrink_batch_only_ever_lowers_the_cap(self, client):
        endpoint = client._endpoints[0]
        endpoint.shrink_batch(100)
        assert endpoint.max_batch == 50
        endpoint.shrink_batch(200)
        assert endpoint.max_batch == 50, "a larger rejection must not raise the cap"
        endpoint.shrink_batch(10)
        assert endpoint.max_batch == 5


class TestMethodRefusal:
    def test_a_refused_method_is_routed_elsewhere_not_retried(self, client):
        """Flashbots is an MEV relay: it whitelists a few methods and rejects
        eth_call. Read as a batch-size problem this splits forever; the fix is
        to remember the refusal per endpoint and route the method away."""
        refuser, good = client._endpoints
        attempts: list[str] = []

        def fake_post(payload, batch_size=1):
            endpoint = client._next_endpoint(batch_size, client._methods_in(payload))
            attempts.append(endpoint.url)
            if endpoint is refuser:
                body = [
                    {"jsonrpc": "2.0", "id": i["id"],
                     "error": {"code": -32601, "message": "rpc method is not whitelisted"}}
                    for i in payload
                ]
                from dex_data.rpc import _method_refusal
                if _method_refusal(body) is not None:
                    endpoint.mark_unsupported(client._methods_in(payload))
                    return good, [
                        {"jsonrpc": "2.0", "id": i["id"], "result": "0x1"} for i in payload
                    ]
            return endpoint, [
                {"jsonrpc": "2.0", "id": i["id"], "result": "0x1"} for i in payload
            ]

        client._post = fake_post
        for _ in range(4):
            assert client.batch([("eth_call", [{}, "latest"])] * 3) == ["0x1"] * 3

        assert "eth_call" in refuser.unsupported
        # After the first refusal every later call must skip the refuser.
        assert attempts[1:] == [good.url] * 3

    def test_refusal_does_not_disable_the_endpoint_for_other_methods(self, client):
        endpoint = client._endpoints[0]
        endpoint.mark_unsupported(["eth_call"])

        assert not endpoint.supports(["eth_call"])
        assert endpoint.supports(["eth_getLogs"])
        assert endpoint.available, "a method refusal is not a cooldown"

    def test_method_refusal_is_not_confused_with_a_revert(self):
        from dex_data.rpc import RpcError

        assert RpcError("rpc method is not whitelisted").is_method_unsupported
        assert not RpcError("execution reverted").is_method_unsupported

    def test_mixed_batch_results_are_not_read_as_a_method_refusal(self, client):
        """Only a uniform failure says anything about the endpoint."""
        from dex_data.rpc import _method_refusal

        body = [
            {"id": 1, "result": "0x1"},
            {"id": 2, "error": {"code": -32601, "message": "rpc method is not whitelisted"}},
        ]
        assert _method_refusal(body) is None

    def test_getlogs_refusal_leaves_the_endpoint_available_for_eth_call(self, client):
        """publicnode rejects unfiltered eth_getLogs but serves large eth_call
        batches better than anyone. Retiring the endpoint over the getLogs
        refusal cost us exactly the endpoint pool verification needed most."""
        from dex_data.rpc import RpcError, _endpoint_fault, _method_refusal

        message = ("Please specify an address in your request or, to remove "
                   "restrictions, order a dedicated full node")
        body = {"error": {"code": -32701, "message": message}}

        assert RpcError(message).is_method_unsupported
        assert _method_refusal(body) is not None
        assert _endpoint_fault(body) is None, "must not retire the whole endpoint"

        endpoint = client._endpoints[0]
        endpoint.mark_unsupported(["eth_getLogs"])
        assert endpoint.available
        assert endpoint.supports(["eth_call"])


class TestRangeLimitWordings:
    """Every provider phrases its getLogs span limit differently, and an
    unmatched wording means the chunk fails instead of splitting. Each of these
    was seen from a live endpoint."""

    @pytest.mark.parametrize("message", [
        "query returned more than 10000 results",
        "block range is too large",
        "Block range too large",
        "exceed maximum block range",
        "Under the Free tier plan, you can make eth_getLogs requests with up to "
        "a 10 block range. Upgrade to PAYG for expanded block range.",
        "Log response size exceeded",
        "range is too large",
    ])
    def test_span_limits_trigger_a_split(self, message):
        assert RpcError(message).is_range_too_large

    def test_a_span_limit_never_retires_the_endpoint(self):
        from dex_data.rpc import _endpoint_fault

        message = ("Under the Free tier plan, you can make eth_getLogs requests "
                   "with up to a 10 block range")
        assert _endpoint_fault({"error": {"code": -32600, "message": message}}) is None

    def test_get_logs_adaptive_halves_until_the_range_is_accepted(self, client):
        """The behaviour the wordings feed into: keep halving, never give up on
        a range that is merely too wide."""
        accepted_spans: list[int] = []

        def fake_get_logs(from_block, to_block, topics=None, address=None):
            span = to_block - from_block + 1
            if span > 10:
                raise RpcError("up to a 10 block range")
            accepted_spans.append(span)
            return [{"blockNumber": hex(from_block)}] * span

        client.get_logs = fake_get_logs
        logs = client.get_logs_adaptive(1000, 1039)

        assert len(logs) == 40, "no block may be dropped by the splitting"
        assert max(accepted_spans) <= 10


class TestAdaptiveThrottling:
    """Free tiers meter compute units per second, not requests, so the safe
    request rate depends on how expensive the calls are and cannot be known up
    front. With a single configured endpoint there is nowhere to fail over to,
    so a 429 has to mean 'go slower', not 'stand this endpoint down'."""

    def test_a_429_halves_the_rate_and_success_eases_it_back(self, monkeypatch):
        monkeypatch.setenv("DEX_RPC_URLS", "https://only.example")
        monkeypatch.setenv("DEX_RPC_RATE_LIMIT", "8")
        client = RpcClient(Config())
        endpoint = client._endpoints[0]

        assert endpoint.limiter.rate == pytest.approx(8.0)
        assert endpoint.throttle() == pytest.approx(4.0)
        assert endpoint.throttle() == pytest.approx(2.0)

        for _ in range(25):
            endpoint.succeeded()
        assert endpoint.limiter.rate > 2.0, "a clean stretch should recover rate"

    def test_the_rate_never_reaches_zero(self, monkeypatch):
        monkeypatch.setenv("DEX_RPC_URLS", "https://only.example")
        monkeypatch.setenv("DEX_RPC_RATE_LIMIT", "8")
        endpoint = RpcClient(Config())._endpoints[0]
        for _ in range(50):
            endpoint.throttle()
        assert endpoint.limiter.rate >= endpoint.limiter.MIN_RATE

    def test_recovery_never_exceeds_the_configured_rate(self, monkeypatch):
        monkeypatch.setenv("DEX_RPC_URLS", "https://only.example")
        monkeypatch.setenv("DEX_RPC_RATE_LIMIT", "8")
        endpoint = RpcClient(Config())._endpoints[0]
        for _ in range(500):
            endpoint.succeeded()
        assert endpoint.limiter.rate == pytest.approx(8.0)


class TestLearnedLogSpan:
    def test_a_discovered_span_cap_is_reused_on_later_calls(self, client):
        """Alchemy's free tier caps eth_getLogs at 10 blocks. Rediscovering
        that by trial and rejection on every chunk wastes a request each
        time, so the cap is remembered."""
        rejected: list[int] = []

        def fake_get_logs(from_block, to_block, topics=None, address=None):
            span = to_block - from_block + 1
            if span > 10:
                rejected.append(span)
                raise RpcError("up to a 10 block range")
            return [{"blockNumber": hex(from_block)}] * span

        client.get_logs = fake_get_logs

        assert len(client.get_logs_adaptive(0, 39)) == 40
        first_round = len(rejected)
        assert first_round > 0, "the first chunk has to discover the cap"

        assert len(client.get_logs_adaptive(100, 139)) == 40
        assert len(rejected) == first_round, "the cap must not be rediscovered"

    def test_the_cap_only_ever_tightens(self, client):
        client._note_span_limit(40)
        assert client._max_log_span == 20
        client._note_span_limit(100)
        assert client._max_log_span == 20, "a looser observation must not widen it"
        client._note_span_limit(10)
        assert client._max_log_span == 5


class TestStandDown:
    """Retiring an endpoint for ten minutes is only sensible when there is
    somewhere else to send the work. With one configured endpoint -- the
    recommended Alchemy setup -- it stalls the entire run, so a refusal has to
    mean 'go slower' instead."""

    def _client(self, monkeypatch, urls):
        monkeypatch.setenv("DEX_RPC_URLS", urls)
        monkeypatch.setenv("DEX_RPC_RATE_LIMIT", "8")
        return RpcClient(Config())

    def test_the_only_endpoint_is_throttled_not_retired(self, monkeypatch):
        client = self._client(monkeypatch, "https://only.example")
        endpoint = client._endpoints[0]

        client._stand_down(endpoint, "quota exceeded")

        assert endpoint.limiter.rate == pytest.approx(4.0), "should slow down"
        assert endpoint.cooldown_until - time.monotonic() < 10, (
            "must not be stood down for ten minutes"
        )

    def test_one_of_several_endpoints_is_retired(self, monkeypatch):
        client = self._client(monkeypatch, "https://a.example,https://b.example")
        dead = client._endpoints[0]

        client._stand_down(dead, "quota exceeded")

        assert not dead.available
        assert dead.cooldown_until - time.monotonic() > 300

    def test_an_endpoint_is_not_an_alternative_for_a_method_it_refuses(self, monkeypatch):
        client = self._client(monkeypatch, "https://a.example,https://b.example")
        a, b = client._endpoints
        b.mark_unsupported(["eth_call"])

        # For eth_call, `a` is effectively alone, so it must be throttled and
        # kept, not stood down for ten minutes.
        client._stand_down(a, "quota exceeded", ["eth_call"])
        assert a.limiter.rate == pytest.approx(4.0)
        assert a.cooldown_until - time.monotonic() < 10


class TestAimdRecovery:
    """Multiplicative decrease, additive increase -- the TCP shape. A
    proportional increase recovers in a number of steps that grows with the
    configured ceiling; additive does not, which is what makes recovery fit
    inside a short run."""

    def _endpoint(self, monkeypatch, rate):
        monkeypatch.setenv("DEX_RPC_URLS", "https://only.example")
        monkeypatch.setenv("DEX_RPC_RATE_LIMIT", str(rate))
        return RpcClient(Config())._endpoints[0]

    def test_recovery_takes_ten_steps_whatever_the_ceiling(self, monkeypatch):
        for ceiling in (4, 8, 20):
            endpoint = self._endpoint(monkeypatch, ceiling)
            while endpoint.limiter.rate > endpoint.limiter.MIN_RATE:
                endpoint.throttle()

            steps = 0
            while endpoint.limiter.rate < ceiling - 1e-9:
                endpoint.limiter.relax()
                steps += 1
            assert steps <= 10, f"ceiling {ceiling} took {steps} steps"

    def test_ten_successes_are_enough_for_one_step(self, monkeypatch):
        endpoint = self._endpoint(monkeypatch, 8)
        endpoint.throttle()                      # 8 -> 4
        assert endpoint.limiter.rate == pytest.approx(4.0)

        for _ in range(10):
            endpoint.succeeded()
        assert endpoint.limiter.rate == pytest.approx(4.8)  # +8/10

    def test_a_failure_resets_the_success_streak(self, monkeypatch):
        endpoint = self._endpoint(monkeypatch, 8)
        endpoint.throttle()
        for _ in range(9):
            endpoint.succeeded()
        endpoint.throttle()                      # streak must restart
        for _ in range(9):
            endpoint.succeeded()
        assert endpoint.limiter.rate == pytest.approx(2.0), "no step from a broken streak"

    def test_recovery_still_stops_at_the_configured_rate(self, monkeypatch):
        endpoint = self._endpoint(monkeypatch, 8)
        for _ in range(1000):
            endpoint.succeeded()
        assert endpoint.limiter.rate == pytest.approx(8.0)
