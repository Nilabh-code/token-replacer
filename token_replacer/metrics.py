"""Counters. Deliberately tiny — a prometheus client is not worth a dependency
until someone is actually scraping this."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Metrics:
    served_local: int = 0
    served_upstream: int = 0
    escalated: int = 0
    local_failures: int = 0
    upstream_failures: int = 0
    reason_counts: Counter = field(default_factory=Counter)

    def record_decision(self, tier: str, reason: str) -> None:
        """Routing choice, before anything is known about the outcome."""
        self.reason_counts[reason] += 1

    def record_served(self, tier: str) -> None:
        """Which tier actually delivered the response."""
        if tier == "local":
            self.served_local += 1
        else:
            self.served_upstream += 1

    def record_escalation(self) -> None:
        """The local tier was chosen and could not deliver; upstream took over."""
        self.escalated += 1

    def as_dict(self) -> dict:
        total = self.served_local + self.served_upstream
        return {
            "requests": total,
            "served_local": self.served_local,
            "served_upstream": self.served_upstream,
            "escalated": self.escalated,
            "local_failures": self.local_failures,
            "upstream_failures": self.upstream_failures,
            "local_share": round(self.served_local / total, 4) if total else 0.0,
            "reasons": dict(self.reason_counts),
        }
