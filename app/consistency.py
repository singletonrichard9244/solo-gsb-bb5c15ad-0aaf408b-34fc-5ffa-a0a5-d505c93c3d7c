"""Redundant-channel consistency check (read-only).

Two different existing channels are treated as redundant measurements of the
same quantity.  Each channel's samples are expanded into the half-open pieces
on which a value is actually held — ``[sample_ts, min(next sample,
sample_ts + TTL, horizon))`` — exactly as the rule engine does, and the
sample taken *before* the query window starts is carried in: a piece that is
still valid at ``start`` covers ``start`` as well.

The two piece lists are then swept together and compared only on stretches
where **both** channels hold a value at the same instant — the half-open
intersection of the two coverings.  Wherever one channel's value has expired
(TTL gap) or ends at the global data horizon there is simply nothing to
compare; values are never extrapolated into such stretches.

On every common stretch the absolute difference ``|v1 - v2|`` is constant; a
stretch where it is *strictly greater* than the non-negative threshold forms
an inconsistency.  Adjacent inconsistent stretches that touch end-to-end
(``[a,b)`` followed by ``[b,c)``) are merged into one segment, which reports
its start/end, duration in seconds and the maximum absolute difference inside
it together with the two samples (timestamps and values) at which that
maximum occurs.

The check is a pure function of the supplied held pieces: it never inserts,
updates or deletes anything and never triggers rule re-evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .engine import HeldValue

_EPS = 1e-7  # seconds of slack for boundary comparisons


@dataclass(frozen=True)
class InconsistencySegment:
    """One maximal half-open stretch ``[start, end)`` with |v1 - v2| > threshold.

    ``max_diff`` is the largest absolute difference inside the segment;
    ``channel_1_sample`` / ``channel_2_sample`` identify the two held samples
    at which that maximum occurs (timestamp + value + event id when known).
    """

    start: float
    end: float
    max_diff: float
    channel_1_sample: dict
    channel_2_sample: dict

    @property
    def duration_seconds(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "duration_seconds": self.end - self.start,
            "max_diff": self.max_diff,
            "channel_1_sample": dict(self.channel_1_sample),
            "channel_2_sample": dict(self.channel_2_sample),
        }


@dataclass(frozen=True)
class ConsistencyResult:
    start: float
    end: float
    threshold: float
    comparable_seconds: float
    inconsistent_seconds: float
    segments: list[InconsistencySegment] = field(default_factory=list)

    @property
    def consistent_seconds(self) -> float:
        return self.comparable_seconds - self.inconsistent_seconds

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "threshold": self.threshold,
            "comparable_seconds": self.comparable_seconds,
            "inconsistent_seconds": self.inconsistent_seconds,
            "consistent_seconds": self.consistent_seconds,
            "inconsistency_rate": (
                self.inconsistent_seconds / self.comparable_seconds
                if self.comparable_seconds > 0
                else 0.0
            ),
            "segment_count": len(self.segments),
            "segments": [seg.to_dict() for seg in self.segments],
        }


def _piece_at(pieces: list[HeldValue], t: float, hint: int) -> tuple[HeldValue | None, int]:
    """Return the held piece covering instant ``t`` and a pointer hint.

    Pieces are disjoint half-open intervals ordered by start; ``hint`` is an
    index at/behind which the piece may be found (the sweep only moves
    forward).  Returns (None, hint) when no piece covers ``t``.
    """
    i = max(0, hint)
    while i < len(pieces) and pieces[i].end <= t + _EPS:
        i += 1
    if i < len(pieces) and pieces[i].start <= t + _EPS and pieces[i].end > t + _EPS:
        return pieces[i], i
    return None, i


def _sample_ref(piece: HeldValue) -> dict:
    return {
        "event_id": piece.event_id,
        "ts": piece.sample_ts,
        "value": piece.value,
    }


def channel_consistency(
    pieces_1: list[HeldValue],
    pieces_2: list[HeldValue],
    start: float,
    end: float,
    threshold: float,
) -> ConsistencyResult:
    """Compare two channels' held values over the half-open window ``[start, end)``.

    ``pieces_1`` / ``pieces_2`` are the channels' held-value pieces expanded
    with their own TTLs and the global data horizon (a piece still valid at
    ``start`` from an earlier sample must be included by the caller).  Only
    time covered by both channels inside the window is comparable; expired
    gaps and the post-horizon stretch are never extrapolated.

    ``threshold`` is a non-negative, finite absolute difference; comparison
    is strictly greater-than (``|v1 - v2| > threshold``).  Touching bad
    stretches merge into one segment.

    Raises ``ValueError`` for an inverted/non-finite window or a negative /
    non-finite threshold.
    """
    start = float(start)
    end = float(end)
    threshold = float(threshold)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("consistency end must be greater than start (half-open [start, end))")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("diff threshold must be a finite non-negative absolute difference")

    # Only pieces that can overlap the window matter; pieces already over at
    # start are skipped via the pointer advancement in _piece_at.
    p1 = sorted(pieces_1, key=lambda p: (p.start, p.end))
    p2 = sorted(pieces_2, key=lambda p: (p.start, p.end))

    # Boundaries inside (start, end) at which either channel's held value can
    # change; the state on [t_i, t_{i+1}) is constant for both channels.
    # start/end are added separately, so only strictly interior instants go
    # in here (epsilon slack absorbs float rounding at the edges).
    bounds = {start, end}
    for pieces in (p1, p2):
        for piece in pieces:
            if piece.start > start + _EPS and piece.start < end - _EPS:
                bounds.add(piece.start)
            if piece.end > start + _EPS and piece.end < end - _EPS:
                bounds.add(piece.end)
    timeline = sorted(bounds)

    comparable_seconds = 0.0
    inconsistent_seconds = 0.0
    raw: list[tuple[float, float, float, dict, dict]] = []
    i1 = i2 = 0
    for idx, t in enumerate(timeline):
        if idx + 1 >= len(timeline):
            break
        t_next = timeline[idx + 1]
        a, i1 = _piece_at(p1, t, i1)
        b, i2 = _piece_at(p2, t, i2)
        if a is None or b is None:
            continue  # one channel expired / no sample yet: not extrapolated
        lo = max(t, start)
        hi = min(t_next, end, a.end, b.end)
        if hi <= lo + _EPS:
            continue
        length = hi - lo
        comparable_seconds += length
        diff = abs(a.value - b.value)
        if diff > threshold + _EPS:
            inconsistent_seconds += length
            raw.append((lo, hi, diff, _sample_ref(a), _sample_ref(b)))

    # Merge touching inconsistent stretches, carrying the maximum difference
    # and the pair of samples that attained it.
    segments: list[InconsistencySegment] = []
    for lo, hi, diff, s1, s2 in raw:
        if segments and lo <= segments[-1].end + _EPS:
            prev = segments[-1]
            if diff + _EPS >= prev.max_diff:
                merged = InconsistencySegment(
                    start=prev.start, end=hi, max_diff=diff,
                    channel_1_sample=s1, channel_2_sample=s2,
                )
            else:
                merged = InconsistencySegment(
                    start=prev.start, end=hi, max_diff=prev.max_diff,
                    channel_1_sample=prev.channel_1_sample,
                    channel_2_sample=prev.channel_2_sample,
                )
            segments[-1] = merged
        else:
            segments.append(
                InconsistencySegment(
                    start=lo, end=hi, max_diff=diff,
                    channel_1_sample=s1, channel_2_sample=s2,
                )
            )

    return ConsistencyResult(
        start=start,
        end=end,
        threshold=threshold,
        comparable_seconds=comparable_seconds,
        inconsistent_seconds=inconsistent_seconds,
        segments=segments,
    )
