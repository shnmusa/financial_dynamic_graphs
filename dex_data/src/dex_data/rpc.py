"""Thread-safe JSON-RPC client with endpoint failover and adaptive log ranges."""
from __future__ import annotations

import itertools
import logging
import random
import threading
import time
from typing import Any, Iterable, Sequence

import requests

from . import abi
from .config import CONFIG, Config

log = logging.getLogger(__name__)

# Sub-calls packed into one Multicall3 eth_call. Each is a cheap view call, so
# the ceiling is calldata size rather than gas; 250 keeps the request around
# 50 KB, which every endpoint tested accepts.
MULTICALL_CHUNK = 250


class RpcError(RuntimeError):
    """A JSON-RPC error object came back from an endpoint."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code

    @property
    def is_endpoint_fault(self) -> bool:
        """Whether the failure is the endpoint's, not the request's.

        A public endpoint that starts demanding an API key, or has run out of
        quota, will answer *every* request this way. Those must fail over to
        another endpoint instead of propagating, or one dead URL in the list
        takes down the whole run.
        """
        text = str(self).lower()
        needles = (
            "unauthorized", "api key", "apikey", "forbidden", "authenticate",
            "exceeded", "capacity", "quota", "credits", "payment required",
            "service unavailable", "too many requests", "rate limit",
        )
        return any(n in text for n in needles)

    @property
    def is_transient(self) -> bool:
        """A transport-level failure rather than a verdict on the request.

        Some providers signal an oversized eth_getLogs response with a bare
        HTTP 500 or 521 instead of a JSON-RPC error, so a caller holding a
        splittable block range should try splitting before giving up.
        """
        text = str(self).lower()
        if self.is_endpoint_fault or self.is_method_unsupported:
            return False
        return any(
            n in text
            for n in ("all rpc attempts failed", "timeout", "timed out",
                      "connection", "500", "502", "503", "520", "521", "524")
        )

    @property
    def is_method_unsupported(self) -> bool:
        """Whether the endpoint refuses this METHOD, not this request.

        Kept apart from is_endpoint_fault because the response differs: a
        method an endpoint will never serve should be routed elsewhere
        permanently, while the endpoint stays usable for everything else.
        Flashbots, for instance, is an MEV relay that whitelists a handful of
        methods and rejects eth_call outright.
        """
        text = str(self).lower()
        return any(
            n in text
            for n in (
                "not whitelisted",
                "method not found",
                "method not supported",
                "method does not exist",
                "unsupported method",
                "is not available",
                # publicnode: refuses unfiltered getLogs, and gates historical
                # ranges behind a paid token. Both are limits on this method,
                # not on the endpoint -- it still serves eth_call batches well.
                "specify an address",
                "dedicated full node",
                "archive requests",
                "personal token",
            )
        )

    @property
    def is_batch_too_large(self) -> bool:
        """Endpoints cap JSON-RPC batch size, and each picks a different cap.

        Splitting the batch is the right response, so this is kept distinct
        from is_endpoint_fault -- the endpoint is healthy, our request was fat.
        """
        text = str(self).lower()
        return any(
            n in text
            for n in (
                "too many rpc calls in batch",
                "batch too large",
                "batch size",
                "exceeds the limit of batch",
                "too many requests in batch",
                # drpc's free plan phrasing
                "batch of more than",
                "requests are not allowed",
            )
        )

    @property
    def is_range_too_large(self) -> bool:
        """Whether the endpoint is telling us to ask for fewer blocks.

        Every provider words this differently and most reuse generic error
        codes, so message matching is unavoidable here.
        """
        text = str(self).lower()
        needles = (
            "more than 10000 results",
            "query returned more than",
            "response size exceed",
            "response too large",
            "log response size exceeded",
            "query timeout exceeded",
            # Providers word their getLogs span limit every possible way --
            # "block range is too large", "up to a 10 block range", "exceed
            # maximum block range". Matching the phrase they all contain is
            # more durable than chasing each wording, and a needless split
            # costs one extra request.
            "block range",
            "range is too large",
            "limit exceeded",
            "too many results",
            "request entity too large",
            "payload too large",
            "-32005",
        )
        return any(n in text for n in needles)


class _RateLimiter:
    """Request spacing for one endpoint, adjustable while the run is going.

    Free tiers meter by compute units per second, not requests, so the right
    request rate depends on how expensive the calls happen to be -- an
    unfiltered eth_getLogs over 20 busy blocks costs far more than a
    getBlockByNumber. There is no way to know the safe rate up front, so it is
    discovered: halve on a 429, ease back up while requests keep succeeding.
    """

    MIN_RATE = 0.5

    def __init__(self, rate_per_sec: float):
        self._max_rate = rate_per_sec
        self._rate = rate_per_sec
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    @property
    def rate(self) -> float:
        return self._rate

    def _interval(self) -> float:
        return 1.0 / self._rate if self._rate > 0 else 0.0

    def acquire(self) -> None:
        with self._lock:
            interval = self._interval()
            if interval <= 0:
                return
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait <= 0:
                self._next_allowed = now + interval
                wait = 0.0
            else:
                self._next_allowed += interval
        if wait > 0:
            time.sleep(wait)

    def throttle(self) -> float:
        with self._lock:
            self._rate = max(self.MIN_RATE, self._rate / 2)
            return self._rate

    def relax(self) -> None:
        """Step back towards the configured rate after a clean stretch.

        Additive increase against the multiplicative decrease in throttle() --
        the same AIMD shape TCP uses, and for the same reason: back off hard
        when the far end complains, then recover in a bounded number of steps.
        A proportional increase does not do that. Recovering from the 0.5/s
        floor to a configured 8/s took ~500 successful requests at 1.15x per
        25; at a tenth of the ceiling per 10 it takes ~100, which is the
        difference between recovering inside a short run and not.
        """
        with self._lock:
            if self._rate < self._max_rate:
                self._rate = min(self._max_rate, self._rate + self._max_rate / 10)


class _Endpoint:
    def __init__(self, url: str, rate_per_sec: float):
        self.url = url
        self.limiter = _RateLimiter(rate_per_sec)
        self._local = threading.local()
        self.lock = threading.Lock()
        # Endpoints that keep failing get skipped until this timestamp.
        self.cooldown_until = 0.0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        # Learned at runtime. Free tiers cap JSON-RPC batch size and every one
        # picks a different cap (drpc allows 3, flashbots a few dozen,
        # publicnode hundreds), and none of them advertise it. So: start
        # optimistic, halve on rejection, and route batches only to endpoints
        # known to accept that size.
        self.max_batch: int | None = None
        # Methods this endpoint has refused outright. Public endpoints expose
        # wildly different method sets and none of them advertise it.
        self.unsupported: set[str] = set()

    def http(self) -> requests.Session:
        # requests.Session is not documented as thread-safe; give each thread
        # its own so connection pooling stays sane under the worker pool.
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            self._local.session = session
        return session

    def penalise(self, seconds: float, escalate: bool = True) -> None:
        """Cool this endpoint off, for longer each time it fails in a row."""
        with self.lock:
            if escalate:
                self.consecutive_failures += 1
                seconds *= min(2 ** (self.consecutive_failures - 1), 8)
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)

    def succeeded(self) -> None:
        with self.lock:
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            recovered = self.consecutive_successes % 10 == 0
        if recovered:
            self.limiter.relax()

    def throttle(self) -> float:
        with self.lock:
            self.consecutive_successes = 0
        return self.limiter.throttle()

    def shrink_batch(self, attempted: int) -> None:
        with self.lock:
            cap = max(1, attempted // 2)
            self.max_batch = cap if self.max_batch is None else min(self.max_batch, cap)

    def accepts_batch(self, size: int) -> bool:
        return size <= 1 or self.max_batch is None or self.max_batch >= size

    def mark_unsupported(self, methods: Iterable[str]) -> None:
        with self.lock:
            self.unsupported.update(methods)

    def supports(self, methods: Sequence[str]) -> bool:
        return not self.unsupported.intersection(methods)

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.cooldown_until


def _first_error(body: Any) -> RpcError | None:
    """Pull one representative error out of a single or batch response."""
    if isinstance(body, dict):
        err = body.get("error")
        items = [err] if err else []
    else:
        items = [i.get("error") for i in (body or []) if isinstance(i, dict) and i.get("error")]
        # For a batch, only a *uniform* failure says something about the
        # endpoint; mixed results are ordinary per-call outcomes.
        if not items or len(items) != len(body or []):
            return None
    if not items:
        return None
    err = items[0]
    message = str(err.get("message", err)) if isinstance(err, dict) else str(err)
    code = err.get("code") if isinstance(err, dict) else None
    return RpcError(message, code)


def _method_refusal(body: Any) -> RpcError | None:
    error = _first_error(body)
    return error if error is not None and error.is_method_unsupported else None


def _endpoint_fault(body: Any) -> RpcError | None:
    """Return the error if `body` is an endpoint-level failure, else None.

    Per-call errors inside a batch are left alone -- a single reverting
    eth_call is not a reason to abandon an endpoint.
    """
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if not err:
        return None
    message = str(err.get("message", err)) if isinstance(err, dict) else str(err)
    code = err.get("code") if isinstance(err, dict) else None
    candidate = RpcError(message, code)
    # Both of these mean "send a smaller request", not "this endpoint is broken".
    if candidate.is_range_too_large or candidate.is_batch_too_large:
        return None
    return candidate if candidate.is_endpoint_fault else None


class RpcClient:
    """Round-robins requests across endpoints and retries with backoff."""

    def __init__(self, config: Config = CONFIG):
        self.config = config
        self._endpoints = [_Endpoint(u, config.rpc_rate_limit) for u in config.rpc_urls]
        self._cycle = itertools.cycle(range(len(self._endpoints)))
        self._cycle_lock = threading.Lock()
        self._id_counter = itertools.count(1)
        # Largest eth_getLogs span accepted so far. Free tiers cap this (10
        # blocks on Alchemy's, for instance) and splitting rediscovers the cap
        # on every chunk unless it is remembered.
        self._max_log_span: int | None = None
        self._span_lock = threading.Lock()

    # -- transport ----------------------------------------------------------

    def _has_alternative(self, endpoint: _Endpoint, methods: Sequence[str] = ()) -> bool:
        """Whether another endpoint could serve this work if we stood one down."""
        return any(
            other is not endpoint and other.supports(methods)
            for other in self._endpoints
        )

    def _stand_down(self, endpoint: _Endpoint, reason: object,
                    methods: Sequence[str] = ()) -> None:
        """Retire an endpoint -- but only if there is somewhere else to go.

        With a single configured endpoint (the recommended Alchemy setup) a ten
        minute retirement stalls the whole run. There the right response to a
        refusal is to go slower and let the caller retry, not to remove the only
        way of reaching the chain.
        """
        if self._has_alternative(endpoint, methods):
            endpoint.penalise(600.0)
            log.warning("retiring %s for 10 min: %s", endpoint.url, reason)
        else:
            rate = endpoint.throttle()
            endpoint.penalise(2.0, escalate=False)
            log.warning("%s refused a request and is the only endpoint; "
                        "slowing to %.2f req/s instead of retiring: %s",
                        endpoint.url, rate, reason)

    def _next_endpoint(self, batch_size: int = 1, methods: Sequence[str] = ()) -> _Endpoint:
        with self._cycle_lock:
            fallback: _Endpoint | None = None
            for _ in range(len(self._endpoints)):
                endpoint = self._endpoints[next(self._cycle)]
                if not endpoint.available or not endpoint.supports(methods):
                    continue
                if endpoint.accepts_batch(batch_size):
                    return endpoint
                fallback = fallback or endpoint
            if fallback is not None:
                return fallback
            # Nothing ideal is free. Prefer an endpoint that at least supports
            # the method over one that has already refused it.
            usable = [e for e in self._endpoints if e.supports(methods)]
            return min(usable or self._endpoints, key=lambda e: e.cooldown_until)

    @staticmethod
    def _methods_in(payload: Any) -> tuple[str, ...]:
        if isinstance(payload, dict):
            return (payload.get("method", ""),)
        return tuple({item.get("method", "") for item in payload or []})

    def _post(self, payload: Any, batch_size: int = 1) -> tuple[_Endpoint, Any]:
        """One JSON-RPC round trip, retried across endpoints.

        Returns the endpoint alongside the body so callers can attribute a
        batch rejection to the endpoint that made it.
        """
        last_error: Exception | None = None
        methods = self._methods_in(payload)

        for attempt in range(self.config.rpc_max_retries):
            endpoint = self._next_endpoint(batch_size, methods)
            endpoint.limiter.acquire()
            try:
                response = endpoint.http().post(
                    endpoint.url, json=payload, timeout=self.config.rpc_timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                endpoint.penalise(2.0)
                time.sleep(min(2**attempt * 0.4, 8) + random.random() * 0.3)
                continue

            if response.status_code == 429:
                rate = endpoint.throttle()
                # Short cooldown only: with a single configured endpoint there
                # is nowhere to fail over to, so the fix is to go slower, not
                # to stand the endpoint down.
                endpoint.penalise(1.0, escalate=False)
                last_error = RpcError(f"429 rate limited by {endpoint.url}")
                log.debug("throttling %s to %.2f req/s", endpoint.url, rate)
                time.sleep(min(2**attempt * 0.6, 10) + random.random() * 0.5)
                continue

            if response.status_code in (413, 414):
                # The request was too big for this endpoint, not malformed.
                endpoint.shrink_batch(batch_size)
                endpoint.succeeded()
                last_error = RpcError(f"{response.status_code} payload too large")
                continue

            if response.status_code >= 500:
                endpoint.penalise(3.0)
                last_error = RpcError(f"{response.status_code} from {endpoint.url}")
                time.sleep(min(2**attempt * 0.4, 8))
                continue

            try:
                body = response.json()
            except ValueError as exc:
                # Some endpoints answer an oversized batch with an HTML error
                # page rather than JSON, so treat it as a size signal too.
                last_error = exc
                if batch_size > 1:
                    endpoint.shrink_batch(batch_size)
                else:
                    endpoint.penalise(1.0)
                continue

            # An endpoint that does not serve this method never will. Record it
            # and route the method elsewhere, keeping the endpoint for others.
            refusal = _method_refusal(body)
            if refusal is not None:
                endpoint.mark_unsupported(methods)
                last_error = refusal
                log.info("%s does not serve %s: %s",
                         endpoint.url, ",".join(methods), refusal)
                continue

            # A JSON-RPC error can still be the endpoint's fault (expired key,
            # exhausted quota). Retire it for a while and try the next one.
            fault = _endpoint_fault(body)
            if fault is not None:
                last_error = fault
                self._stand_down(endpoint, fault, methods)
                continue

            endpoint.succeeded()
            return endpoint, body

        raise RpcError(f"all RPC attempts failed: {last_error}") from last_error

    # -- api ----------------------------------------------------------------

    def call(self, method: str, params: Sequence[Any] | None = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._id_counter),
            "method": method,
            "params": list(params or []),
        }
        _endpoint, body = self._post(payload)
        if isinstance(body, list):
            body = body[0] if body else {}
        if "error" in body and body["error"]:
            err = body["error"]
            raise RpcError(str(err.get("message", err)), err.get("code"))
        return body.get("result")

    def batch(self, calls: Sequence[tuple[str, Sequence[Any]]]) -> list[Any]:
        """Send many calls in one HTTP round trip.

        Results come back positionally. A call that genuinely failed on chain
        (a token with no symbol(), a v2 pool asked for fee()) becomes None --
        that is expected and callers handle it. A call that failed because the
        *endpoint* rejected the batch is retried on a smaller batch elsewhere,
        never silently returned as None: a dropped factory() result would
        quietly discard a real pool and break swap pairing downstream.
        """
        if not calls:
            return []

        payload = []
        id_to_index: dict[int, int] = {}
        for index, (method, params) in enumerate(calls):
            request_id = next(self._id_counter)
            id_to_index[request_id] = index
            payload.append(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": list(params)}
            )

        def split_retry() -> list[Any]:
            mid = len(calls) // 2
            return self.batch(calls[:mid]) + self.batch(calls[mid:])

        try:
            endpoint, body = self._post(payload, batch_size=len(calls))
        except RpcError as exc:
            if (exc.is_batch_too_large or exc.is_range_too_large) and len(calls) > 1:
                return split_retry()
            raise

        # Some endpoints answer an oversized batch with a single top-level error.
        if isinstance(body, dict):
            if body.get("error"):
                err = body["error"]
                error = RpcError(str(err.get("message", err)), err.get("code"))
                if error.is_batch_too_large and len(calls) > 1:
                    endpoint.shrink_batch(len(calls))
                    return split_retry()
                raise error
            body = [body]

        results: list[Any] = [None] * len(calls)
        retry_indices: list[int] = []
        error_messages: list[str] = []

        for item in body or []:
            index = id_to_index.get(item.get("id"))
            if index is None:
                continue
            err = item.get("error")
            if not err:
                results[index] = item.get("result")
                continue
            error = RpcError(
                str(err.get("message", err)) if isinstance(err, dict) else str(err),
                err.get("code") if isinstance(err, dict) else None,
            )
            error_messages.append(str(error))
            if error.is_batch_too_large or error.is_endpoint_fault:
                retry_indices.append(index)
            # Anything else is a real per-call failure; None is the right answer.

        # Every single item failing with the same message is not per-call
        # semantics -- it is the endpoint refusing the batch in a wording we do
        # not recognise. Retry those rather than reporting a wall of Nones.
        if (
            not retry_indices
            and len(calls) > 1
            and len(error_messages) == len(calls)
            and len(set(error_messages)) == 1
        ):
            log.warning(
                "%s rejected a batch of %d with an unrecognised error: %s",
                endpoint.url, len(calls), error_messages[0][:120],
            )
            retry_indices = list(range(len(calls)))

        if retry_indices:
            if len(retry_indices) == len(calls):
                # The endpoint rejected the whole batch item by item -- exactly
                # what drpc's free plan does past 3 requests.
                endpoint.shrink_batch(len(calls))
                if len(calls) > 1:
                    return split_retry()
                self._stand_down(
                    endpoint,
                    f"rejects even single calls: {error_messages[0][:120]}",
                    self._methods_in(payload),
                )
                raise RpcError(
                    f"endpoint {endpoint.url} rejected the request: "
                    f"{error_messages[0][:200]}"
                )

            retry_calls = [calls[i] for i in retry_indices]
            for position, value in zip(retry_indices, self.batch(retry_calls)):
                results[position] = value

        return results

    # -- convenience --------------------------------------------------------

    def block_number(self) -> int:
        return int(self.call("eth_blockNumber"), 16)

    def get_block(self, number: int | str, full_tx: bool = False) -> dict | None:
        tag = number if isinstance(number, str) else hex(number)
        return self.call("eth_getBlockByNumber", [tag, full_tx])

    def get_block_timestamps(self, numbers: Iterable[int], batch_size: int = 25) -> dict[int, int]:
        """Fetch many block timestamps using JSON-RPC batching."""
        numbers = list(dict.fromkeys(numbers))
        out: dict[int, int] = {}
        for start in range(0, len(numbers), batch_size):
            window = numbers[start : start + batch_size]
            calls = [("eth_getBlockByNumber", [hex(n), False]) for n in window]
            for number, result in zip(window, self.batch(calls)):
                if result and result.get("timestamp"):
                    out[number] = int(result["timestamp"], 16)
        return out

    def eth_call(self, to: str, data: str, block: int | str = "latest") -> str | None:
        tag = block if isinstance(block, str) else hex(block)
        return self.call("eth_call", [{"to": to, "data": data}, tag])

    def eth_call_batch(
        self, calls: Sequence[tuple[str, str]], block: int | str = "latest"
    ) -> list[str | None]:
        tag = block if isinstance(block, str) else hex(block)
        return self.batch([("eth_call", [{"to": to, "data": data}, tag]) for to, data in calls])

    def multicall(
        self,
        calls: Sequence[tuple[str, str]],
        block: int | str = "latest",
        chunk_size: int = MULTICALL_CHUNK,
    ) -> list[str | None]:
        """Run many view calls as ONE eth_call each `chunk_size`, via Multicall3.

        This is the single biggest saving against a metered endpoint. Free tiers
        price per JSON-RPC method call, so 6318 separate eth_calls to read pool
        and token metadata saturate the per-second compute budget on their own;
        the same work through Multicall3 is a few dozen requests.

        Falls back to plain per-call batching whenever Multicall3 cannot serve
        the request -- a historical block before it was deployed, or a node that
        rejects the aggregate for size. Results are positional either way, and a
        reverting sub-call is None in both, so callers cannot tell which path ran.
        """
        if not calls:
            return []

        if isinstance(block, int) and block < abi.MULTICALL3_DEPLOY_BLOCK:
            return self.eth_call_batch(calls, block)

        results: list[str | None] = []
        for start in range(0, len(calls), chunk_size):
            window = list(calls[start : start + chunk_size])
            results.extend(self._multicall_chunk(window, block))
        return results

    def _multicall_chunk(
        self, calls: list[tuple[str, str]], block: int | str
    ) -> list[str | None]:
        try:
            raw = self.eth_call(
                abi.MULTICALL3_ADDRESS, abi.encode_aggregate3(calls), block
            )
        except RpcError as exc:
            # Too big for this node, or it does not serve the aggregate. Halve
            # and retry; a single call that still fails falls back to a plain
            # eth_call so one bad target cannot lose the whole window.
            if len(calls) > 1:
                mid = len(calls) // 2
                return (
                    self._multicall_chunk(calls[:mid], block)
                    + self._multicall_chunk(calls[mid:], block)
                )
            log.debug("multicall fell back to eth_call: %s", exc)
            return self.eth_call_batch(calls, block)

        decoded = abi.decode_aggregate3(raw)
        if len(decoded) != len(calls):
            # A truncated or unexpected response must never be read as "these
            # calls reverted" -- a dropped factory() would silently discard a
            # real pool. Redo the window the slow, unambiguous way.
            log.warning(
                "multicall returned %d results for %d calls; falling back",
                len(decoded), len(calls),
            )
            return self.eth_call_batch(calls, block)
        return decoded

    def get_logs(
        self,
        from_block: int,
        to_block: int,
        topics: list[Any] | None = None,
        address: list[str] | str | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {"fromBlock": hex(from_block), "toBlock": hex(to_block)}
        if topics:
            params["topics"] = topics
        if address:
            params["address"] = address
        return self.call("eth_getLogs", [params]) or []

    def _note_span_limit(self, span: int) -> None:
        with self._span_lock:
            cap = max(1, span // 2)
            self._max_log_span = cap if self._max_log_span is None else min(
                self._max_log_span, cap
            )

    def get_logs_adaptive(
        self,
        from_block: int,
        to_block: int,
        topics: list[Any] | None = None,
        address: list[str] | str | None = None,
    ) -> list[dict]:
        """get_logs that halves the block range whenever a provider says it is
        too big, recursively, down to a single block.

        This is what makes a fixed DEX_LOG_CHUNK_BLOCKS safe across providers
        with wildly different result caps -- pick a hint, let it self-tune. The
        discovered cap is remembered, so later chunks start at a span that
        works instead of paying for the same rejection every time.
        """
        with self._span_lock:
            cap = self._max_log_span
        if cap is not None and to_block - from_block + 1 > cap:
            out: list[dict] = []
            for start in range(from_block, to_block + 1, cap):
                out.extend(
                    self.get_logs_adaptive(
                        start, min(start + cap - 1, to_block), topics, address
                    )
                )
            return out

        try:
            return self.get_logs(from_block, to_block, topics, address)
        except RpcError as exc:
            splittable = from_block < to_block
            # Split on an explicit "too many results", and also on a transport
            # failure: providers that answer an oversized range with a bare 500
            # give us no other signal, and halving a range that was merely
            # unlucky costs one extra request.
            if not splittable or not (exc.is_range_too_large or exc.is_transient):
                raise
            if exc.is_range_too_large:
                self._note_span_limit(to_block - from_block + 1)
            mid = from_block + (to_block - from_block) // 2
            log.debug("splitting %d-%d after: %s", from_block, to_block, exc)
            left = self.get_logs_adaptive(from_block, mid, topics, address)
            right = self.get_logs_adaptive(mid + 1, to_block, topics, address)
            return left + right


_client: RpcClient | None = None
_client_lock = threading.Lock()


def get_client() -> RpcClient:
    """Process-wide shared client, so rate limiting is actually global."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = RpcClient()
    return _client
