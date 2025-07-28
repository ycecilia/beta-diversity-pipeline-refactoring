def get_taxonomic_ranks(taxonomic_rank):
    """
    Get the taxonomic ranks and corresponding keys up to the specified rank.
    """
    if taxonomic_rank == "taxonomic_path":
        taxonomic_rank = "species"
    # List of all taxonomic ranks
    all_taxonomic_ranks = [
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]

    # Find the index of taxonomic_rank in the list of all taxonomic ranks
    taxonomic_num = all_taxonomic_ranks.index(taxonomic_rank)

    # Get only the ranks up to and including the specified rank
    taxonomic_ranks = all_taxonomic_ranks[: taxonomic_num + 1]
    taxonomic_key_ranks = [rank + "Key" for rank in taxonomic_ranks]

    return [taxonomic_ranks, taxonomic_key_ranks, taxonomic_num]
