# Schema Map (Fetcher → Analyzer → Ideation)

_Last updated: 2025-10-16 · Timezone: Europe/Berlin · Current versions: Fetcher v4, Analyzer v4, Ideation v1_

This document is the **authoritative data contract** for the AI Agency Sprint pipeline.  
It defines required fields, types, defaults, and edge cases across stages.

---

## 1) Fetcher → Analyzer (Normalized Post Records)

**Minimum required fields (must be present per post):**

| Field | Type | Req. | Description | Example |
|---|---|---|---|---|
| Creator | string | ✅ | Display name of the creator | "Jason Holmes" |
| Platform | enum | ✅ | instagram \| tiktok \| youtube | "youtube" |
| HandleOrChannelId | string | ✅ | Platform handle or channel id | "UCxyz" |
| PostId | string | ✅ | Platform-unique post id | "yt:AbC123" |
| PostURL | string (url) | ✅ | Canonical post url | "https://youtu.be/AbC123" |
| PublishedAt | datetime (ISO-8601, UTC) | ✅ | Original publish time | "2025-10-15T19:22:00Z" |
| CollectedAt | datetime (ISO-8601, UTC) | ✅ | Time fetched by Fetcher | "2025-10-16T08:00:00Z" |
| Text | string|null | ✅ | Caption/description/title (null if absent) | "How to stop tantrums fast…" |
| Metrics.Views | number | ✅ | View count (int) | 720000 |
| Metrics.Likes | number | ✅ | Like count (int) | 54000 |
| Metrics.Comments | number | ✅ | Comment count (int) | 1320 |

**Optional but recommended:**

| Field | Type | Req. | Description | Example |
|---|---|---|---|---|
| HookCandidate | string|null | ◻️ | First 8–12 words or on-image text (if OCR) | "This one thing ends every meltdown…" |
| ThumbnailURL | string|null | ◻️ | Thumbnail image url | "https://i.ytimg.com/.../default.jpg" |
| Tags | array<string> | ◻️ | Hashtags/keywords | ["parenting","tantrums"] |
| Language | string|null | ◻️ | BCP-47 code if reliably known | "en" |
| SentimentHint | enum|null | ◻️ | Positive \| Neutral \| Negative | "Positive" |
| EngagementPriorityScore | number | ◻️ | Views*1.0 + Likes*0.1 + Comments*0.2 | 726000.0 |
| Provenance.Connector | enum | ◻️ | youtube_api \| instagram_graph \| tiktok_api \| rss \| csv | "youtube_api" |
| Provenance.Cursor | string|null | ◻️ | Paging token | "PAGE_TOKEN_2" |
| Provenance.FetchedWith | enum | ◻️ | api \| export \| feed | "api" |
| Provenance.SinceWindowDays | number | ◻️ | Window used for fetch | 7 |

**Constraints & Defaults**  
- Coerce numeric fields; missing metrics → default 0.  
- PublishedAt ≤ CollectedAt ≤ now(UTC).  
- Deduplicate by (Platform, PostId).  
- If comments unavailable: SentimentHint = "Neutral".  
- If Text missing and no safe substitute, **skip** record (Analyzer requires it).

**Edge Cases**  
- Non-English content: set Language if reliable; otherwise null.  
- Private/removed posts: exclude and log WARN.  
- OCR for HookCandidate is optional and may require Vision API (see README).

---

## 2) Analyzer → Ideation (Summaries & Idea Seeds)

**Analyzer output fields:**

| Field | Type | Req. | Description |
|---|---|---|---|
| Creator | string | ✅ | From Fetcher |
| Platform | enum | ✅ | From Fetcher |
| PostURL | string | ✅ | From Fetcher |
| PostDate | datetime | ✅ | Copy of PublishedAt |
| Topic | string | ✅ | Core idea summary |
| HookLine | string | ✅ | ≤ 90 chars |
| Likes | number | ✅ | From Fetcher |
| Views | number | ✅ | From Fetcher |
| Comments | number | ✅ | From Fetcher |
| EngagementScore | number | ✅ | Views*0.6 + Likes*0.3 + Comments*0.1 |
| CommentSentiment | enum | ✅ | Positive \| Neutral \| Negative |
| HookArchetype | string | ✅ | e.g., "curiosity gap" |
| Angle | string | ✅ | emotional \| logical \| practical |
| Takeaways | array<string> | ✅ | 3–5 bullets |
| ProcessedAt | datetime | ✅ | When Analyzer ran |
| SourceRef | string | ✅ | Platform:PostId or URL |

**Ideation input fields:** same as Analyzer output, consumed to produce idea cards.

**Ideation output fields:**

| Field | Type | Req. | Description |
|---|---|---|---|
| IdeaTitle | string | ✅ | Headline idea |
| OneLiner | string | ✅ | 1–2 sentence pitch |
| SuggestedHook | string | ✅ | Short hook suggestion |
| CTA | string | ✅ | Call-to-action |
| Hashtags | array<string> | ◻️ | Relevant tags |
| FormatHints | enum | ◻️ | Reel \| Short \| Carousel |
| ConfidenceScore | number (0–1) | ◻️ | Heuristic |
| GeneratedAt | datetime | ✅ | Timestamp |
| SourcePostURL | string | ✅ | Back-reference |

---

## 3) Versioning & Change Policy

- Current: Fetcher v4, Analyzer v4, Ideation v1.  
- Breaking changes (MAJOR): rename/remove fields → bump major and update this file.  
- Additive changes (MINOR): add optional fields only.  
- Non-functional tweaks (PATCH): copy edits/clarifications.

Tag releases in Git, e.g., `analyzer-v4.0.0`, `fetcher-v4.0.0`.  
See `/docs/CHANGELOG.md` for notes.

---

## 4) Validation Checklist

- [ ] Sheet tab `analysis` headers match: `RunId,Creator,Platform,PostURL,PostDate,Topic,HookLine,Likes,Views,Comments,EngagementScore,CommentSentiment,HookArchetype,Angle,Takeaways,ProcessedAt,SourceRef`  
- [ ] Sheet tab `ideas` headers match: `IdeaTitle,OneLiner,SuggestedHook,CTA,Hashtags,FormatHints,ConfidenceScore,Creator,Platform,Topic,HookArchetype,Angle,GeneratedAt,SourcePostURL`  
- [ ] Make blueprints point to correct Drive folders.  
- [ ] Analyzer TOP_N=3; Fetcher MAX_POSTS_PER_CREATOR=20; Window=7 days.  
- [ ] OCR (optional) only on top-N thumbnails to control cost.
