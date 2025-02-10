from ai_service.model import Lyrics, Prediction

def rerank_lyrics(candidates: list[Lyrics], original_prediction: Prediction | None = None) -> list[Lyrics]:
    """
    Reranks a list of candidate lyrics, prioritizing artist diversity
    and including one "exploratory" item.
    
    Args:
        candidates: A list of Lyrics objects to be reranked, assumed to be sorted by initial similarity.
        original_prediction: The original prediction that led to these candidates. 
                             (Currently not used for this version of exploration but kept for future enhancements).

    Returns:
        A list of Lyrics objects, reranked.
    """
    if not candidates:
        return []

    print(f"Reranking {len(candidates)} candidates. Original prediction: {original_prediction}")

    # Pass 1: Artist Diversity
    # Items are added to this list if their artist hasn't been seen yet.
    # Since 'candidates' is sorted by similarity, this picks the most similar song for each new artist.
    diverse_items_list = []
    seen_artists = set()
    for lyric in candidates:
        if lyric.artist not in seen_artists:
            diverse_items_list.append(lyric)
            seen_artists.add(lyric.artist)

    # Pass 2: Prepare remaining candidates (those not picked for diversity in the first pass)
    # These are still in their original similarity-sorted order relative to each other.
    remaining_candidates_after_diversity = [
        lyric for lyric in candidates if lyric not in diverse_items_list
    ]

    # Pass 3: Guided Exploration
    # Promote one less similar item (if available) from the pool of remaining candidates.
    exploratory_item = None
    if remaining_candidates_after_diversity:
        # Take the least similar item from the *remaining* pool to be the exploratory item.
        # .pop(-1) removes and returns the last item.
        exploratory_item = remaining_candidates_after_diversity.pop(-1) 

    # Combine the lists:
    # 1. Items selected for artist diversity (most similar per artist).
    # 2. The single exploratory item (which was the least similar among the remaining, now promoted).
    # 3. The rest of the candidates (in their original similarity order, minus the exploratory item).
    
    final_reranked_list = list(diverse_items_list) # Start with diverse items

    if exploratory_item:
        final_reranked_list.append(exploratory_item) # Add the promoted exploratory item

    # Add the rest of the remaining candidates (which no longer includes the exploratory_item as it was popped)
    final_reranked_list.extend(remaining_candidates_after_diversity)
    
    # The list construction should inherently avoid duplicates if Lyrics objects are distinct
    # and the logic correctly partitions and reassembles them.

    print(f"Returning {len(final_reranked_list)} reranked candidates after diversity and exploration pass.")
    return final_reranked_list

