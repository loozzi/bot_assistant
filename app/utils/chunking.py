def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks, sized in approximate tokens (~4 chars/token).

    Word-count based, not fixed character-count splitting, so chunk boundaries
    fall on word edges. Overlap is clamped below the chunk size to guarantee
    forward progress regardless of the chunk_size/overlap values passed in.
    """
    words = text.split()
    if not words:
        return [text]

    chunk_word_size = max(chunk_size // 4, 1)
    overlap_words = min(max(overlap // 4, 0), chunk_word_size - 1)

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_word_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words

    return chunks
