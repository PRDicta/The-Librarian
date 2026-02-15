"""
The Librarian — Query Expander

Transforms a raw user query into multiple search variants for broader recall.
Three capabilities:

1. **Synonym expansion**: Maps intent words ("struggling", "broke") to search
   terms that match how entries are actually stored ("error", "fix", "debug").

2. **Intent detection**: Identifies whether the query is experiential
   ("what was I struggling with"), factual ("how does X work"), or
   retrospective ("what did we decide about Y") and biases search accordingly.

3. **Category routing**: When experiential intent is detected, returns
   category hints so the searcher can weight entries tagged as corrections,
   friction, breakthroughs, etc.

No LLM calls — this is entirely heuristic and runs in microseconds.
"""
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from .entity_extractor import EntityExtractor, ExtractedEntities


# ─── Intent Types ────────────────────────────────────────────────────────────

class QueryIntent:
    EXPERIENTIAL = "experiential"    # "what did I struggle with", "where did I get stuck"
    FACTUAL = "factual"              # "how does X work", "what is Y"
    RETROSPECTIVE = "retrospective"  # "what did we decide", "what was the plan"
    EXPLORATORY = "exploratory"      # default — broad search


@dataclass
class ExpandedQuery:
    """Result of query expansion — multiple search variants + metadata."""
    original: str
    variants: List[str]              # All query variants to search
    intent: str = QueryIntent.EXPLORATORY
    category_bias: List[str] = field(default_factory=list)  # Categories to boost
    category_filter: List[str] = field(default_factory=list)  # Categories to restrict to (if strong signal)
    entities: Optional[ExtractedEntities] = None  # Phase 11: Extracted entities for re-ranking


# ─── Synonym / Intent Maps ──────────────────────────────────────────────────

# Maps user intent words → search terms that match stored entries
EXPERIENTIAL_SYNONYMS = {
    # Resolution / solution
    "solution": ["fix", "resolved", "approach", "implementation", "workaround", "answer"],
    "solved": ["fix", "solution", "resolved", "working", "answer"],
    "answer": ["solution", "fix", "resolved", "result"],
    # Frustration / struggle
    "struggling": ["error", "fix", "debug", "wrong", "failed", "broken", "issue"],
    "struggled": ["error", "fix", "debug", "wrong", "failed", "broken", "issue"],
    "stuck": ["error", "fix", "debug", "blocked", "issue", "workaround"],
    "frustrated": ["error", "fix", "wrong", "failed", "retry", "broken"],
    "confused": ["clarified", "corrected", "misunderstood", "actually"],
    "broke": ["error", "fix", "broken", "regression", "failed"],
    "failed": ["error", "fix", "failure", "wrong", "retry"],
    "wrong": ["corrected", "fix", "actually", "mistake", "wrong command"],
    "mistake": ["corrected", "fix", "actually", "wrong"],
    "problem": ["error", "fix", "issue", "debug", "workaround"],
    "issue": ["error", "fix", "bug", "debug", "problem"],
    "hard": ["difficult", "complex", "challenge", "workaround"],
    "difficult": ["complex", "challenge", "workaround", "hard"],
    # Success / breakthrough
    "breakthrough": ["solved", "fix", "working", "success", "resolved"],
    "solved": ["fix", "solution", "resolved", "working"],
    "figured out": ["solution", "resolved", "discovery", "realized"],
    "realized": ["corrected", "actually", "discovery", "understood"],
    "eureka": ["solved", "fix", "breakthrough", "working"],
    # Structure / organization
    "organizing": ["hierarchy", "structure", "consolidation", "grouping", "taxonomy", "architecture"],
    "organized": ["hierarchy", "structure", "consolidated", "grouped", "categorized"],
    "structure": ["hierarchy", "architecture", "organization", "layout", "design"],
    # Change of direction
    "pivoted": ["changed", "switched", "instead", "decided", "replaced"],
    "changed": ["switched", "replaced", "updated", "modified", "instead"],
    "switched": ["changed", "replaced", "migrated", "moved to"],
    "abandoned": ["replaced", "removed", "dropped", "instead"],
}

RETROSPECTIVE_SYNONYMS = {
    "decided": ["decision", "chose", "agreed", "plan", "went with"],
    "decision": ["decided", "chose", "agreed", "plan"],
    "chose": ["decision", "selected", "went with", "picked"],
    "plan": ["decided", "strategy", "approach", "design"],
    "agreed": ["decided", "consensus", "plan", "approach"],
}

# Domain / technical synonyms — bidirectional term bridging.
# These run on every query regardless of intent, catching cases where
# the query uses different jargon than the stored entry.
# Each key maps to terms that should also be searched.
TECHNICAL_SYNONYMS = {
    # DevOps / deployment
    "ci/cd": ["deployment pipeline", "continuous integration", "continuous deployment", "build and deploy", "github actions", "ci cd"],
    "ci cd": ["ci/cd", "deployment pipeline", "continuous integration", "continuous deployment", "build and deploy"],
    "deployment pipeline": ["ci/cd", "deploy", "shipping", "release", "github actions", "build pipeline"],
    "deployment": ["deploy", "release", "shipping", "ci/cd", "pipeline"],
    "deploy": ["deployment", "release", "shipping", "ci/cd", "pipeline"],
    "shipping code": ["deployment", "deploy", "release", "ci/cd", "pipeline"],
    "shipping": ["deployment", "deploy", "release", "ci/cd"],
    "github actions": ["ci/cd", "deployment pipeline", "workflow", "automation"],
    "docker": ["container", "containerization", "dockerfile", "image"],
    "container": ["docker", "containerization", "kubernetes", "k8s"],
    "kubernetes": ["k8s", "container orchestration", "docker", "cluster"],
    "k8s": ["kubernetes", "container orchestration", "cluster"],
    # Infrastructure
    "aws": ["amazon web services", "cloud", "ec2", "s3", "lambda", "ecs", "fargate"],
    "fargate": ["ecs", "aws", "serverless", "container"],
    "ecs": ["fargate", "aws", "container service", "docker"],
    "serverless": ["lambda", "functions", "faas", "cloud functions"],
    "lambda": ["serverless", "functions", "aws lambda"],
    # Databases
    "database": ["db", "storage", "data layer", "persistence"],
    "db": ["database", "storage", "data layer"],
    "sql": ["database", "relational", "postgres", "mysql", "sqlite"],
    "nosql": ["mongodb", "dynamodb", "document store", "non-relational"],
    "postgres": ["postgresql", "sql", "relational database"],
    "postgresql": ["postgres", "sql", "relational database"],
    "mongodb": ["mongo", "nosql", "document database"],
    "mongo": ["mongodb", "nosql", "document database"],
    "redis": ["cache", "key-value", "in-memory"],
    # Auth
    "authentication": ["auth", "login", "sign in", "identity", "jwt", "oauth"],
    "auth": ["authentication", "authorization", "login", "jwt", "oauth"],
    "jwt": ["json web token", "auth", "authentication", "token"],
    "oauth": ["authentication", "auth", "sso", "single sign-on"],
    "sso": ["single sign-on", "oauth", "authentication"],
    # Frontend
    "frontend": ["front-end", "ui", "client-side", "react", "browser"],
    "front-end": ["frontend", "ui", "client-side"],
    "backend": ["back-end", "server-side", "api", "server"],
    "back-end": ["backend", "server-side", "api"],
    "api": ["endpoint", "rest", "graphql", "backend", "interface"],
    "rest": ["restful", "api", "http", "endpoint"],
    "graphql": ["api", "query language", "schema"],
    # Testing
    "testing": ["tests", "test", "unit test", "integration test", "qa"],
    "tests": ["testing", "test suite", "unit test", "spec"],
    "unit test": ["testing", "jest", "pytest", "spec"],
    "integration test": ["testing", "e2e", "end-to-end"],
    "e2e": ["end-to-end", "integration test", "cypress", "playwright"],
    # Monitoring
    "monitoring": ["observability", "alerting", "logging", "metrics", "sentry", "datadog"],
    "observability": ["monitoring", "logging", "tracing", "metrics"],
    "logging": ["logs", "monitoring", "observability"],
    "alerting": ["alerts", "monitoring", "notifications", "pagerduty"],
    # Architecture
    "microservices": ["micro-services", "service-oriented", "distributed"],
    "monolith": ["monolithic", "single service"],
    "architecture": ["design", "structure", "system design", "patterns"],
    # Version control
    "git": ["version control", "source control", "github", "repository"],
    "github": ["git", "repository", "version control", "pull request"],
    "pull request": ["pr", "code review", "merge request"],
    "pr": ["pull request", "code review", "merge request"],
    # General dev
    "refactor": ["refactoring", "restructure", "clean up", "rewrite"],
    "refactoring": ["refactor", "restructure", "clean up"],
    "performance": ["optimization", "speed", "latency", "throughput"],
    "optimization": ["performance", "optimize", "speed", "efficiency"],
    "caching": ["cache", "redis", "memoization", "cdn"],
    "cache": ["caching", "redis", "memoization", "cdn"],
}

# Phrases that signal experiential intent
EXPERIENTIAL_PHRASES = [
    r"struggl\w*", r"stuck\b", r"frustrat\w*", r"confus\w*",
    r"broke\b", r"broken\b", r"fail\w*", r"wrong\b",
    r"mistake\w*", r"problem\w*", r"issue\w*", r"hard\b",
    r"difficult\w*", r"couldn't\b", r"didn't work",
    r"went wrong", r"messed up", r"screwed up",
    r"pain\w*", r"annoying", r"headache",
]

# Phrases that signal retrospective intent
RETROSPECTIVE_PHRASES = [
    r"decid\w*", r"chose\b", r"chosen\b", r"agree\w*",
    r"plan\w*", r"approach\b", r"strategy\b",
    r"went with\b", r"settled on\b",
]

# Phrases that signal factual intent
FACTUAL_PHRASES = [
    r"how does\b", r"how do\b", r"what is\b", r"what are\b",
    r"explain\b", r"describe\b", r"definition\b",
    r"how .+ work", r"what .+ mean",
]

# Category biases per intent type
INTENT_CATEGORY_BIAS = {
    QueryIntent.EXPERIENTIAL: ["correction", "friction", "breakthrough", "pivot", "warning"],
    QueryIntent.RETROSPECTIVE: ["decision", "preference", "note"],
    QueryIntent.FACTUAL: ["definition", "implementation", "fact", "reference"],
    QueryIntent.EXPLORATORY: [],  # No bias — search everything
}


# ─── Query Expander ──────────────────────────────────────────────────────────

class QueryExpander:
    """
    Expands a raw query into multiple search variants with intent metadata.
    Phase 11: Now includes entity extraction for exact-match boosting.
    Purely heuristic — no LLM calls, runs in microseconds.
    """

    def __init__(self):
        self._entity_extractor = EntityExtractor()

    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query into variants with intent detection.

        Returns an ExpandedQuery with:
        - variants: list of search strings (always includes the original)
        - intent: detected intent type
        - category_bias: categories to boost in results
        - entities: extracted entities for re-ranking
        """
        lower = query.lower().strip()

        # Step 1: Detect intent
        intent = self._detect_intent(lower)

        # Step 2: Extract entities from the query
        entities = self._entity_extractor.extract_from_query(query)

        # Step 3: Generate synonym-expanded variants
        variants = self._expand_synonyms(lower, intent)

        # Step 4: Generate entity-focused variant
        # If entities were found, create a variant that's just the entities
        # This is what makes "Philip's analogy about books" findable
        if entities.all_entities:
            entity_variant = " ".join(entities.all_entities)
            if entity_variant not in variants:
                variants.append(entity_variant)
            # Also try entities + topic words (stripped query)
            stripped = self._strip_filler(lower)
            if stripped:
                entity_topic = " ".join(entities.all_entities) + " " + stripped
                if entity_topic not in variants:
                    variants.append(entity_topic)

        # Step 5: Generate a keyword-stripped variant (remove filler words)
        stripped = self._strip_filler(lower)
        if stripped and stripped != lower and stripped not in variants:
            variants.append(stripped)

        # Always include the original query
        if query not in variants:
            variants.insert(0, query)

        # Step 6: Get category bias based on intent
        category_bias = INTENT_CATEGORY_BIAS.get(intent, [])

        return ExpandedQuery(
            original=query,
            variants=variants[:10],  # Cap at 10 — entity + technical synonym variants need room
            intent=intent,
            category_bias=category_bias,
            entities=entities,
        )

    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query from phrase patterns."""
        # Check experiential first (strongest signal)
        for pattern in EXPERIENTIAL_PHRASES:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryIntent.EXPERIENTIAL

        # Check retrospective
        for pattern in RETROSPECTIVE_PHRASES:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryIntent.RETROSPECTIVE

        # Check factual
        for pattern in FACTUAL_PHRASES:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryIntent.FACTUAL

        return QueryIntent.EXPLORATORY

    def _expand_synonyms(self, query: str, intent: str) -> List[str]:
        """Generate synonym-expanded query variants."""
        variants = [query]

        # Choose synonym map based on intent
        if intent == QueryIntent.EXPERIENTIAL:
            syn_map = EXPERIENTIAL_SYNONYMS
        elif intent == QueryIntent.RETROSPECTIVE:
            syn_map = RETROSPECTIVE_SYNONYMS
        else:
            syn_map = {**EXPERIENTIAL_SYNONYMS, **RETROSPECTIVE_SYNONYMS}

        # Find matching synonym keys in the query
        expansions: Set[str] = set()
        for trigger, replacements in syn_map.items():
            if trigger in query:
                expansions.update(replacements)

        # Build expanded variant: original topic words + synonym terms
        if expansions:
            # Extract the "topic" part of the query (remove intent words)
            topic_words = self._extract_topic_words(query, syn_map.keys())
            if topic_words:
                # Variant 1: topic + top synonyms
                top_syns = sorted(expansions)[:4]
                variants.append(f"{topic_words} {' '.join(top_syns)}")
                # Variant 2: just the synonyms (broader search)
                variants.append(" ".join(sorted(expansions)[:5]))
            else:
                # No clear topic — just use synonyms
                variants.append(" ".join(sorted(expansions)[:5]))

        # ── Technical synonym expansion (runs on every query) ──
        # Bridges domain jargon: "CI/CD" ↔ "deployment pipeline", etc.
        # Uses word-boundary matching to avoid false positives
        # ("pr" in "preferences" should NOT trigger "pull request").
        tech_expansions: Set[str] = set()
        matched_triggers: List[str] = []
        for trigger, replacements in TECHNICAL_SYNONYMS.items():
            if self._match_technical_term(trigger, query):
                tech_expansions.update(replacements)
                matched_triggers.append(trigger)

        if tech_expansions:
            # Remove matched triggers from query to isolate remaining context
            topic_words = self._strip_matched_triggers(query, matched_triggers)
            top_tech = sorted(tech_expansions, key=len)[:4]  # Prefer shorter terms
            if topic_words:
                variants.append(f"{topic_words} {' '.join(top_tech)}")
            # Also add a pure technical synonym variant
            variants.append(" ".join(sorted(tech_expansions, key=len)[:5]))

        return variants

    @staticmethod
    def _match_technical_term(trigger: str, query: str) -> bool:
        """Check if a technical trigger matches as a whole word/phrase in the query.

        Handles special characters like '/' in 'ci/cd' by escaping them
        for regex, then requiring word boundaries (or string edges).
        """
        escaped = re.escape(trigger)
        # Use word boundaries, but also allow / as a boundary character
        pattern = r'(?:^|(?<=\s)|(?<=/))' + escaped + r'(?:$|(?=\s)|(?=/))'
        return bool(re.search(pattern, query, re.IGNORECASE))

    @staticmethod
    def _strip_matched_triggers(query: str, triggers: List[str]) -> str:
        """Remove matched technical triggers from query to isolate remaining context."""
        result = query
        # Sort by length descending to strip longest phrases first
        for trigger in sorted(triggers, key=len, reverse=True):
            escaped = re.escape(trigger)
            result = re.sub(escaped, " ", result, flags=re.IGNORECASE)
        return " ".join(result.split()).strip()

    def _extract_topic_words(self, query: str, intent_words) -> str:
        """Extract the meaningful topic words, stripping intent/filler words."""
        words = query.split()
        filler = set(intent_words) | _QUERY_FILLER
        topic = [w for w in words if w.lower() not in filler]
        return " ".join(topic).strip()

    def _strip_filler(self, query: str) -> str:
        """Strip common filler words to get a tighter keyword query.

        Preserves compound terms containing '/' (like 'ci/cd') as single tokens
        so they aren't broken apart during word-level filtering.
        """
        words = query.split()
        stripped = [w for w in words if w.lower() not in _QUERY_FILLER or "/" in w]
        return " ".join(stripped).strip()


# Common filler words to remove for tighter keyword searches
_QUERY_FILLER = {
    "i", "me", "my", "we", "our", "the", "a", "an", "is", "was", "were",
    "am", "are", "been", "be", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    "what", "which", "where", "when", "how", "who", "whom",
    "that", "this", "these", "those", "it", "its",
    "with", "about", "for", "from", "in", "on", "at", "to", "of", "by",
    "and", "or", "but", "not", "so", "if", "then",
    "there", "here", "just", "also", "very", "really", "quite",
}
