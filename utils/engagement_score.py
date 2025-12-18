def normalize_engagement(likes, comments, views):
    raw = 0.5 * likes + 0.3 * comments + 0.2 * views
    return raw / (raw + 1e-6)
