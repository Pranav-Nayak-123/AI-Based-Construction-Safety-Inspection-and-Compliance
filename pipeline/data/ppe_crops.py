"""Person crops with head/torso PPE labels derived from box annotations (v7 §34.5–34.6).

A PPE box is assigned to the person whose box contains its centre, in the body region the
item belongs to (helmets in the top of the box, vests in the middle). Only *explicit*
annotations become labels: a person with no head annotation gets no head label (ignored
in training), never an assumed negative. `unknown` is taught separately, from crops where
the region is demonstrably not visible (see the training augmentations).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

HEAD_LABELS = ("helmet", "no_helmet", "unknown")
TORSO_LABELS = ("vest", "no_vest", "unknown")
IGNORE = -100

# v7 §34.6: 10 % horizontal, 5 % top, 3 % bottom padding.
PAD_X, PAD_TOP, PAD_BOTTOM = 0.10, 0.05, 0.03
HEAD_REGION = (0.0, 0.40)  # fraction of person height from the top
TORSO_REGION = (0.12, 0.80)


class HeadLabel(IntEnum):
    HELMET = 0
    NO_HELMET = 1
    UNKNOWN = 2


class TorsoLabel(IntEnum):
    VEST = 0
    NO_VEST = 1
    UNKNOWN = 2


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def padded(self, width: int, height: int) -> Box:
        return Box(
            max(0.0, self.x1 - PAD_X * self.width),
            max(0.0, self.y1 - PAD_TOP * self.height),
            min(float(width), self.x2 + PAD_X * self.width),
            min(float(height), self.y2 + PAD_BOTTOM * self.height),
        )


@dataclass(frozen=True)
class PersonLabels:
    person: Box
    head: int  # HeadLabel value or IGNORE
    torso: int  # TorsoLabel value or IGNORE


def _owner(item: Box, persons: Sequence[Box], region: tuple[float, float]) -> int | None:
    """Index of the smallest person box holding `item`'s centre inside `region`."""
    cx, cy = item.center
    best: int | None = None
    for index, person in enumerate(persons):
        margin = 0.10 * person.width
        if not (person.x1 - margin <= cx <= person.x2 + margin):
            continue
        top = person.y1 + region[0] * person.height
        bottom = person.y1 + region[1] * person.height
        if not (top <= cy <= bottom):
            continue
        if best is None or person.area < persons[best].area:
            best = index
    return best


def associate(
    persons: Sequence[Box],
    helmets: Sequence[Box],
    no_helmets: Sequence[Box],
    vests: Sequence[Box],
    no_vests: Sequence[Box],
) -> list[PersonLabels]:
    """Per-person head/torso labels; conflicting evidence on a region → ignored."""
    head: list[set[int]] = [set() for _ in persons]
    torso: list[set[int]] = [set() for _ in persons]
    for boxes, label, votes, region in (
        (helmets, HeadLabel.HELMET, head, HEAD_REGION),
        (no_helmets, HeadLabel.NO_HELMET, head, HEAD_REGION),
        (vests, TorsoLabel.VEST, torso, TORSO_REGION),
        (no_vests, TorsoLabel.NO_VEST, torso, TORSO_REGION),
    ):
        for box in boxes:
            owner = _owner(box, persons, region)
            if owner is not None:
                votes[owner].add(int(label))

    def resolve(votes: set[int]) -> int:
        return next(iter(votes)) if len(votes) == 1 else IGNORE

    return [
        PersonLabels(person, resolve(head[i]), resolve(torso[i]))
        for i, person in enumerate(persons)
    ]
