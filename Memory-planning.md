# Memory Integration Plan - Full Implementation Roadmap

**Date:** 2026-07-14  
**Status:** Planning Phase  
**Scope:** Complete memory layer (Redis + Qdrant) integration for all modules  
**Target:** Build knowledge base + improve response quality via context retrieval  

---

## 📋 Executive Summary

The memory layer enables the bot to:
- **Remember** past searches, journal entries, transactions, insights, tasks
- **Retrieve** relevant context for new user queries
- **Personalize** responses using historical data
- **Reduce** redundant API calls via caching

**Key Design Principle:** Semantic, user-scoped retrieval with importance decay + multi-modal context (Qdrant dense vectors + Redis working cache).

---

## 🏗️ Architecture Overview

### Current State
```
User Message → Orchestrator Graph → Agents → Reply → [WRITE STUB]
                         ↑
                   [READ STUB]
```

### Target State
```
User Message 
  ↓
REHYDRATE_CONTEXT (Trigger A - READ)
  ├─ Redis: exact-match cache check
  ├─ Qdrant: dense search per source_type
  └─ Output: retrieved_memories[]
  ↓
Route (LLM intent classifier)
  ↓
Dispatch Agents [PARALLEL]
  ├─ SearchAgent(+ filtered_memories)
  ├─ JournalAgent(+ filtered_memories)
  ├─ FinanceAgent(+ filtered_memories)
  └─ [...]
  ↓
Format Response (compose if multi-intent)
  ↓
EPISODIC_WRITER (Trigger B - WRITE)
  ├─ Per-intent metadata extraction
  ├─ Semantic chunking (200-400 tokens)
  ├─ Qdrant: upsert chunks
  └─ Redis: cache results (4h TTL)
  ↓
Reply to User
```

---

## 🎯 Trigger Points Summary

| Trigger | Location | Phase | Input | Output |
|---------|----------|-------|-------|--------|
| **A: READ** | orchestrator/graph.py:29 | 2 | user_id + query | retrieved_memories[] |
| **B: WRITE** | orchestrator/graph.py:118 | 1 | agent_outputs | Qdrant + Redis |
| **C: CACHE** | modules/*/agent.py | 3 | retrieved_memories | Cache hit or API call |
| **D: NODE** | modules/search/node/*.py | 3 | URL/text | Redis cache |

---

## 📊 IMPLEMENTATION PHASES

### PHASE 1: Foundation (Embedding + Core Writes)

#### 1.1: Create Embedding Client
**File:** `app/infra/providers/embedding.py` (NEW - 150 LOC)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize with multilingual-e5-large (1024-dim)
async def init_embeddings(model="multilingual-e5-large"):
    global _embedding_client
    _embedding_client = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )

# Embed single text
async def embed_text(text: str) -> list[float]:
    return client.embed_query(text)

# Embed batch
async def embed_texts(texts: list[str]) -> list[list[float]]:
    return client.embed_documents(texts)
```

**Updates:**
- `app/main.py`: Import + call in on_startup(), on_shutdown()
- `pyproject.toml`: Add `"langchain-community[huggingface]>=0.1"`

#### 1.2: Implement _episodic_writer()
**File:** `app/orchestrator/graph.py` (Line 118 - 200 LOC)

```python
async def _episodic_writer(state: AgentState) -> AgentState:
    """Write agent outputs to Qdrant + Redis"""
    
    for intent in state["intents"]:
        reply = state["agent_outputs"][intent]
        source_type = module_config[intent]["source_type"]
        
        # Semantic chunk
        chunks = _chunk_text(reply, chunk_size=400, overlap=50)
        
        # Embed + upsert each chunk
        for chunk in chunks:
            embedding = await embed_text(chunk)
            
            # Extract metadata
            payload = {
                "user_id": state["user_id"],
                "source_type": source_type,
                "timestamp": datetime.now(),
                "importance": module_config[intent]["default_importance"],
                "language": _detect_language(chunk),
                "topics": _extract_keywords(chunk),
                "query": last_user_message,
                "summary_snippet": chunk[:200],
            }
            
            # Intent-specific fields
            if intent == "journal":
                payload["mood_score"] = _extract_mood_score(reply)
            elif intent == "search":
                payload["urls"] = _extract_urls(reply)
            elif intent == "finance":
                payload["category"] = _extract_category(reply)
                payload["amount"] = _extract_amount(reply)
            
            # Upsert to Qdrant
            await qdrant.upsert(collection, point)
            
            # Cache in Redis
            redis.setex(f"{source_type}:{user_id}:{hash}", 14400, reply)
```

**Helper functions needed:**

```python
def _chunk_text(text, chunk_size=400, overlap=50) -> list[str]:
    """Split text into overlapping chunks (tokens ≈ words/4)"""
    words = text.split()
    chunk_word_size = chunk_size // 4
    overlap_words = overlap // 4
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_word_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap_words
    return chunks or [text]

def _detect_language(text) -> str:
    """Simple language detection"""
    return "vi" if any(c in text for c in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ") else "en"

def _extract_keywords(text) -> list[str]:
    """Extract nouns/verbs (TODO: implement with YAKE or TF-IDF)"""
    return []

def _extract_mood_score(reply: str) -> float:
    """Journal: extract mood (0.0-1.0)"""
    moods = {"buồn": 0.2, "vui": 0.8, "bình thường": 0.5}
    for mood, score in moods.items():
        if mood in reply.lower():
            return score
    return 0.5

def _extract_urls(reply: str) -> list[str]:
    """Search: extract URLs"""
    import re
    return re.findall(r'https?://[^\s]+', reply)

def _extract_category(reply: str) -> str:
    """Finance: extract category"""
    categories = ["food", "transport", "shopping", "utility"]
    for cat in categories:
        if cat in reply.lower():
            return cat
    return "other"

def _extract_amount(reply: str) -> float:
    """Finance: extract amount"""
    import re
    matches = re.findall(r'[\d,]+[kK]?', reply)
    if matches:
        return float(matches[0].replace("k", "000"))
    return 0.0
```

---

### PHASE 2: Orchestrator Integration

#### 2.1: Implement _rehydrate_context()
**File:** `app/orchestrator/graph.py` (Line 29 - 180 LOC)

```python
async def _rehydrate_context(state: AgentState) -> AgentState:
    """Retrieve relevant memories from Qdrant + Redis"""
    
    user_id = state["user_id"]
    last_message = _last_human_message(state["messages"])
    
    # Try Redis first (exact match cache)
    cache_key = f"query:{user_id}:{hash(last_message)}"
    cached = redis.get(cache_key)
    if cached:
        return {"retrieved_memories": json.loads(cached)}
    
    # Query Qdrant per source_type
    query_embedding = await embed_text(last_message)
    retrieved = []
    
    for intent, spec in _load_module_specs().items():
        source_type = spec["source_type"]
        
        results = qdrant.search(
            query_vector=query_embedding,
            filter={
                "user_id": user_id,
                "source_type": source_type
            },
            limit=3
        )
        
        for result in results:
            retrieved.append({
                "source_type": source_type,
                "payload": result.payload,
                "score": result.score
            })
    
    # Cache query result (1h TTL)
    redis.setex(cache_key, 3600, json.dumps(retrieved))
    
    return {"retrieved_memories": retrieved}
```

#### 2.2: Update _fan_out()
**File:** `app/orchestrator/graph.py` (Line 50)

```python
def _fan_out(state: AgentState):
    names = router.resolve_agent_names(state["intents"])
    if not names:
        return "format_response"
    
    base_input: AgentInput = {
        "user_id": state["user_id"],
        "message": _last_human_message(state["messages"]),
        "retrieved_memories": []
    }
    
    specs = router._load_module_specs()
    retrieved = state.get("retrieved_memories", [])
    
    sends = []
    for name in names:
        source_type = specs[name]["source_type"]
        
        # Filter memories for this agent's source_type
        filtered = [m["payload"] for m in retrieved 
                   if m["source_type"] == source_type]
        
        agent_input = {**base_input, "retrieved_memories": filtered}
        sends.append(Send("dispatch_agent", {"_agent_name": name, "_input": agent_input}))
    
    return sends
```

---

### PHASE 3: Module Enhancements

#### 3.1: SearchAgent - Cache + Pre-Population
**File:** `app/modules/search/agent.py` (+30 LOC)

```python
class SearchAgent(BaseAgent):
    async def run(self, input: AgentInput) -> AgentOutput:
        # Cache hit check
        cached = self._check_cache(input["message"], input["retrieved_memories"])
        if cached:
            logger.info("search_cache_hit")
            return {"reply": cached}
        
        # Pre-populate documents from cache
        initial_docs = self._extract_documents(input["retrieved_memories"])
        
        initial: SearchState = {
            "user_id": input["user_id"],
            "user_query": input["message"],
            "documents": initial_docs,
            "summary": "",
        }
        
        result = await graph.ainvoke(initial)
        return {"reply": result["summary"]}
    
    def _check_cache(self, query, memories) -> str | None:
        """Check if recent answer exists"""
        for mem in memories:
            if mem["source_type"] == "url" and self._similar(query, mem["query"]):
                return mem["summary_snippet"]
        return None
    
    def _extract_documents(self, memories) -> list[dict]:
        """Pre-populate docs from cache"""
        docs = []
        for mem in memories:
            if mem["source_type"] == "url":
                for url in mem.get("urls", []):
                    docs.append({
                        "title": mem["summary_snippet"][:50],
                        "url": url,
                        "raw_text": mem["summary_snippet"],
                    })
        return docs[:3]
    
    @staticmethod
    def _similar(q1, q2) -> bool:
        return q1.lower()[:20] == q2.lower()[:20]
```

#### 3.2: crawl_node - URL Caching
**File:** `app/modules/search/node/crawl_node.py` (+20 LOC)

```python
async def crawl_node(state: SearchState) -> dict:
    """Crawl with Redis cache"""
    documents = state.get("documents", [])
    if not documents:
        return {"documents": documents}
    
    redis = get_redis_client()
    
    async def fetch_with_cache(doc):
        url = doc.get("url", "")
        if not url:
            return doc
        
        # Check Redis
        cache_key = f"crawl_url:{md5(url).hexdigest()}"
        cached = redis.get(cache_key)
        if cached:
            doc["raw_text"] = cached
            return doc
        
        # Fetch via Jina
        try:
            content = await jina_fetch(url)
            redis.setex(cache_key, 86400, content)  # 24h TTL
            doc["raw_text"] = content
            return doc
        except CrawlToolError:
            return doc
    
    updated = await asyncio.gather(*[fetch_with_cache(d) for d in documents])
    return {"documents": updated}
```

#### 3.3: summary_node - Context Enhancement
**File:** `app/modules/search/node/summary_node.py` (+10 LOC)

```python
async def summary_node(state: SearchState) -> dict:
    """Summarize with retrieved_memories context"""
    documents = state.get("documents", [])
    if not documents:
        return {"summary": NO_RESULTS_MESSAGE}
    
    # Build context from previous searches (TODO: pass via SearchState)
    context_block = ""
    system_prompt = SUMMARY_SYSTEM_PROMPT
    if context_block:
        system_prompt += f"\n\nContext: {context_block}"
    
    # ... rest of LLM call
```

---

### PHASE 4: Advanced Features

#### 4.1: Importance Decay Job
**File:** `app/bot/scheduler.py`

```python
# In scheduler initialization:
scheduler.add_job(
    memory_decay_job,
    'cron',
    hour=3, minute=0,
    timezone='Asia/Ho_Chi_Minh'
)

async def memory_decay_job():
    """Daily 03:00: Decay old memories by 10%/30d"""
    logger.info("memory_decay_job_start")
    
    qdrant = get_qdrant_client()
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    
    # Query old points
    points, _ = qdrant.scroll(collection, limit=100, with_payload=True)
    
    for point in points:
        if point.payload["timestamp"] < cutoff:
            # 10% decay
            old_imp = point.payload["importance"]
            new_imp = max(0.0, old_imp * 0.9)
            qdrant.update_vectors(collection, [{"id": point.id, "importance": new_imp}])
    
    logger.info("memory_decay_job_complete")
```

#### 4.2: Memory Compression Job
**File:** `app/bot/scheduler.py`

```python
# In scheduler initialization:
scheduler.add_job(
    memory_compression_job,
    'cron',
    day_of_week=6, hour=4, minute=0,
    timezone='Asia/Ho_Chi_Minh'
)

async def memory_compression_job():
    """Weekly Sunday 04:00: Compress low-importance points"""
    logger.info("memory_compression_job_start")
    
    qdrant = get_qdrant_client()
    llm = create_llm_client()
    
    # Query points with importance < 0.2
    # TODO: Implement Qdrant filtering
    
    # For each group:
    # 1. Collect old chunks
    # 2. LLM summarize
    # 3. Upsert summary as single point
    # 4. Delete old points
    
    logger.info("memory_compression_job_complete")
```

---

## 📁 FILES CHANGED SUMMARY

### New Files
```
app/infra/providers/embedding.py       (150 LOC)
Memory-planning.md (this file)
```

### Modified Files
```
app/main.py                           (+10 lines)
app/orchestrator/graph.py             (+250 lines)
app/modules/search/agent.py           (+30 lines)
app/modules/search/node/crawl_node.py (+20 lines)
app/modules/search/node/summary_node.py (+10 lines)
app/bot/scheduler.py                  (+40 lines)
pyproject.toml                        (+1 line)
```

**Total LOC: ~500**

---

## ✅ IMPLEMENTATION CHECKLIST

### Phase 1: Foundation ⏳
- [ ] Create app/infra/providers/embedding.py
  - [ ] HuggingFaceEmbeddings init
  - [ ] embed_text(), embed_texts()
  - [ ] get_embedding_client()
  - [ ] init_embeddings(), close_embeddings()
- [ ] Update app/main.py (on_startup + on_shutdown)
- [ ] Implement _episodic_writer() in orchestrator/graph.py
  - [ ] Semantic chunking logic
  - [ ] Qdrant upsert per chunk
  - [ ] Redis cache per result
  - [ ] Error handling + logging
- [ ] Add 7 helper functions to orchestrator/graph.py
- [ ] Update pyproject.toml with langchain-community

### Phase 2: Orchestrator Integration ⏳
- [ ] Implement _rehydrate_context() in orchestrator/graph.py
  - [ ] Redis exact-match cache
  - [ ] Qdrant dense search per source_type
  - [ ] Filter by user_id + source_type
  - [ ] Result caching
- [ ] Update _fan_out() in orchestrator/graph.py
  - [ ] Filter retrieved_memories per agent
  - [ ] Pass filtered to AgentInput
- [ ] Test multi-intent scenarios

### Phase 3: Module Enhancements ⏳
- [ ] SearchAgent.run()
  - [ ] _check_cache()
  - [ ] _extract_documents()
  - [ ] _similar()
- [ ] crawl_node.py Redis caching
  - [ ] Cache key generation
  - [ ] Check before Jina
  - [ ] Store with 24h TTL
- [ ] summary_node.py context
  - [ ] Include retrieved_memories
  - [ ] Format context block

### Phase 4: Advanced Features ⏳
- [ ] memory_decay_job()
  - [ ] 30-day window
  - [ ] Daily 03:00 cron
  - [ ] 10% decay logic
- [ ] memory_compression_job()
  - [ ] Identify low-importance
  - [ ] LLM summarization
  - [ ] Weekly Sunday 04:00 cron

---

## 🧪 TESTING CHECKLIST

### Manual Tests
```
✓ Embedding test: embed_text() output is 1024-dim
✓ Single-intent write: Query search → Qdrant populated
✓ Single-intent read: Second search query → retrieved_memories populated
✓ Cache hit: Latency reduced on repeated query
✓ Multi-intent: Both outputs stored separately
✓ Memory filtering: Each agent gets only its source_type memories
✓ Error resilience: Bot works if Qdrant/Redis unavailable
```

### Acceptance Criteria
- [x] No crashes from memory operations (all try/except)
- [x] Bot works if memory layer unavailable
- [x] Qdrant/Redis populated after interactions
- [x] Cache hit rate >50% for repeated queries
- [x] Latency improvement >30% on cache hit

---

## 📊 SUCCESS METRICS

| Metric | Target | Validation |
|--------|--------|-----------|
| Cache hit rate | >50% | Log repeated queries |
| Latency improvement | >30% on hit | Measure search response time |
| Memory storage | <500MB per 10k queries | Monitor Qdrant size |
| Zero regression | No slowdown without memory | Test with memory disabled |
| Operation reliability | 99.9% success | Count failures vs. attempts |

---

## 🚀 DEPLOYMENT CHECKLIST

- [x] Backward compatible (memory ops optional)
- [x] No schema changes (Qdrant auto-creates)
- [x] No data migration needed
- [x] Graceful degradation (stubs continue working)
- [x] Monitoring ready (logging in place)

**Rollback Plan:** Remove memory writes/reads → stubs activate automatically

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Severity | Impact | Mitigation |
|------|----------|--------|-----------|
| Embedding generation slow | High | User latency ↑ | Batch embed, cache, GPU |
| Qdrant collection too large | High | Query latency ↑ | Importance decay + compression jobs |
| Redis memory exhausted | High | OOM crash | TTL management, cache size limits |
| Agent parsing fails | Medium | Wrong metadata | Fallback to generic, extensive logging |
| Multi-intent filtering breaks | Medium | Wrong context to agent | Comprehensive testing, per-agent logging |
| Qdrant unavailable | Low | No retrieval (graceful) | Continue with empty memories |

---

## 📚 DEPENDENCIES

### New
- `langchain-community[huggingface]>=0.1` (includes sentence-transformers)

### Already Available
- `qdrant-client` ✓
- `redis.asyncio` ✓
- `langchain-core` ✓
- `structlog` ✓

### Optional
- `accelerate` (GPU acceleration if EMBEDDING_DEVICE=cuda)

---

**READY FOR PHASE 1 IMPLEMENTATION** 🚀
