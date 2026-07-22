# AI Brain Architecture

```text
Student survey data
    +
Watch history and likes
    +
Curated AI video/session metadata
    |
    v
[Supervised ML recommender]
    - Learns which videos a student may like.
    - Uses text features from profile and content metadata.
    - Produces a like probability.

[Deep learning concept layer]
    - Represents advanced AI topics through patterns and analogies.
    - Explains concepts at a high-school level.
    - Can later be replaced with a safe LLM or RAG system.

[Reinforcement learning feedback loop]
    - Student likes/dislikes become reward signals.
    - The app updates topic weights.
    - Future recommendations shift toward helpful topics.
```

## Scoring formula in the prototype

```text
recommendation_score =
    0.45 * supervised_like_probability
  + 0.25 * content_similarity
  + 0.15 * topic_match
  + 0.10 * freshness_score
  + 0.05 * feedback_learning_score
```

This is intentionally transparent so students can see how the recommendation system works.
