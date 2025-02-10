import pytest

from ai_service.model import Lyrics, Prediction
from ai_service.reranking import rerank_lyrics

# Sample prediction for use in tests
PRED = Prediction(0.1, 0.2, 0.3, 0.4)

def test_rerank_empty_and_single_candidates():
    """Tests reranking with empty or single candidate lists."""
    assert rerank_lyrics([]) == []
    
    single_candidate = [Lyrics("Artist A", "Title 1", PRED)]
    assert rerank_lyrics(single_candidate, PRED) == single_candidate

def test_rerank_main_diversity_and_exploration_scenario():
    """Tests the primary logic for artist diversity and guided exploration."""
    candidates = [
        Lyrics("Artist A", "A_Song1_Sim0.9", PRED),  # Most similar by Artist A
        Lyrics("Artist B", "B_Song1_Sim0.8", PRED),  # Most similar by Artist B
        Lyrics("Artist A", "A_Song2_Sim0.7", PRED),  # Second by Artist A
        Lyrics("Artist C", "C_Song1_Sim0.6", PRED),  # Most similar by Artist C
        Lyrics("Artist B", "B_Song2_Sim0.5", PRED),  # Second by Artist B
    ]
    
    # Expected order after reranking:
    # 1. A_Song1_Sim0.9 (Diverse A)
    # 2. B_Song1_Sim0.8 (Diverse B)
    # 3. C_Song1_Sim0.6 (Diverse C)
    # 4. B_Song2_Sim0.5 (Exploratory - last from remaining [A_Song2, B_Song2])
    # 5. A_Song2_Sim0.7 (Remaining)
    reranked = rerank_lyrics(candidates, PRED)
    
    assert len(reranked) == 5
    assert [item.title for item in reranked] == [
        "A_Song1_Sim0.9",
        "B_Song1_Sim0.8",
        "C_Song1_Sim0.6",
        "B_Song2_Sim0.5",  # Exploratory item
        "A_Song2_Sim0.7",  # Remaining item
    ]
