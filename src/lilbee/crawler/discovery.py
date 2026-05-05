"""URL discovery: build backend-neutral concurrency and filter specs from ``cfg``."""

from __future__ import annotations

from lilbee.core.config import cfg
from lilbee.crawler.models import ConcurrencySpec, FilterSpec


def build_concurrency_spec() -> ConcurrencySpec:
    """Snapshot the crawl-concurrency settings from ``cfg`` into a spec."""
    return ConcurrencySpec(
        semaphore_count=cfg.crawl_concurrent_requests,
        mean_delay=cfg.crawl_mean_delay,
        max_delay_range=cfg.crawl_max_delay_range,
        retry_on_rate_limit=cfg.crawl_retry_on_rate_limit,
        retry_base_delay_min=cfg.crawl_retry_base_delay_min,
        retry_base_delay_max=cfg.crawl_retry_base_delay_max,
        retry_max_backoff=cfg.crawl_retry_max_backoff,
        retry_max_attempts=cfg.crawl_retry_max_attempts,
    )


def build_filter_spec(*, include_subdomains: bool) -> FilterSpec:
    """Snapshot the filter settings from ``cfg`` + caller flags."""
    return FilterSpec(
        exclude_patterns=list(cfg.crawl_exclude_patterns),
        include_subdomains=include_subdomains,
    )
