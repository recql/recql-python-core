import pytest
from recql.execute.merge import Candidate
from recql.reorder import DiversityReorderer


class DummyStep:
    def __init__(self, strength=0.5, diversity_attributes=None, max_diversity_candidates=1000):
        self.strength = strength
        self.diversity_attributes = diversity_attributes
        self.max_diversity_candidates = max_diversity_candidates


@pytest.mark.asyncio
async def test_diversity_reorderer_basic():
    cands = [
        Candidate(id="1", retrieval_score=1.0, attributes={"genre": "Action"}),
        Candidate(id="2", retrieval_score=0.9, attributes={"genre": "Action"}),
        Candidate(id="3", retrieval_score=0.8, attributes={"genre": "Comedy"}),
    ]
    reorderer = DiversityReorderer()
    # With high diversity strength (0.9), item 3 (Comedy) should be picked before item 2 (Action duplicate)
    res = await reorderer.apply(DummyStep(strength=0.9), cands, {})
    ids = [c.id for c in res]
    assert ids[0] == "1"
    assert ids[1] == "3"
    assert ids[2] == "2"


@pytest.mark.asyncio
async def test_diversity_reorderer_empty():
    reorderer = DiversityReorderer()
    assert await reorderer.apply(DummyStep(), [], {}) == []


@pytest.mark.asyncio
async def test_diversity_reorderer_specific_attributes():
    cands = [
        Candidate(id="1", retrieval_score=1.0, attributes={"genre": "Action", "director": "X"}),
        Candidate(id="2", retrieval_score=0.9, attributes={"genre": "Comedy", "director": "X"}),
        Candidate(id="3", retrieval_score=0.8, attributes={"genre": "Action", "director": "Y"}),
    ]
    reorderer = DiversityReorderer()
    # If diversifying only on 'genre', item 2 (Comedy) is different from item 1 (Action)
    res = await reorderer.apply(DummyStep(strength=0.8, diversity_attributes=["genre"]), cands, {})
    ids = [c.id for c in res]
    assert ids[0] == "1"
    assert ids[1] == "2"
