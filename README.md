─────────────────────────────────────────────────────────────
📂 FETCHER v4 → (raw data)
─────────────────────────────────────────────────────────────
Core fields → Required by Analyzer

• Creator (string)
• Platform (instagram|tiktok|youtube)
• HandleOrChannelId (string)
• PostId (string)
• PostURL (string)
• PublishedAt (ISO-8601 UTC)
• Text (caption/description)
• HookCandidate (first line / auto-derived)
• Metrics.Views / Likes / Comments / Shares (numbers)
• ThumbnailURL (string)
• Tags[] (array)
• Language (string|null)
• SentimentHint (string|null)
• EngagementPriorityScore (float)
• Provenance.* (Connector / Cursor / FetchedWith / SinceWindowDays)

─────────────────────────────────────────────────────────────
📂 ANALYZER v4 → (processed summary)
─────────────────────────────────────────────────────────────
Uses Fetcher fields + adds:

• Topic (core idea)
• HookLine (text hook candidate)
• EngagementScore (weighted metric)
• CommentSentiment (Positive|Neutral|Negative)
• HookArchetype (pattern label)
• Angle (emotional / logical / practical)
• Takeaways[] (short bullet list)
• RunId / ProcessedAt
• SourceRefs (PostId, URL, Creator)

─────────────────────────────────────────────────────────────
📂 IDEATION v4 → (content ideas)
─────────────────────────────────────────────────────────────
Uses Analyzer fields + adds:

• IdeaTitle
• OneLiner
• SuggestedHook
• CTA
• Hashtags[]
• FormatHints (Reel|Short|Carousel)
• ConfidenceScore (0–1)
• GeneratedAt
─────────────────────────────────────────────────────────────