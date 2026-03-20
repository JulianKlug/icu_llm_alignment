"""LLM-based classification for task types and subspecialties using Ollama."""

import hashlib
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple

# Valid categories
VALID_TASK_TYPES = ["Diagnosis", "Prognosis", "Treatment", "Knowledge", "Other"]
VALID_SUBSPECIALTIES = [
    "Cardiovascular", "Respiratory", "Neurological",
    "Infectious Disease", "General surgical", "General medical"
]

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b"
OLLAMA_TIMEOUT = 120
MAX_RETRIES = 2

# Cache path
CACHE_PATH = Path(__file__).parent / ".classification_cache.json"

# Prompt version — bump this to invalidate cache when prompt changes
_PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a medical classification expert. Classify ICU (Intensive Care Unit) questions along two axes.

## Task Types
- **Diagnosis**: Questions asking to identify a condition, disease, cause, or etiology. Includes differential diagnosis and identifying underlying problems.
- **Prognosis**: Questions about expected outcomes, survival, mortality risk, long-term recovery, or disease trajectory.
- **Treatment**: Questions about therapeutic interventions, management plans, medications, dosing, procedures, ventilator settings, fluid management, or next clinical steps.
- **Knowledge**: Questions asking to explain mechanisms, pathophysiology, definitions, guidelines, or criteria. Conceptual or educational questions.
- **Other**: Questions that do not fit any of the above categories.

## Subspecialties
- **Cardiovascular**: Heart, hemodynamics, cardiac arrest, arrhythmias, shock (cardiogenic/distributive), vasopressors, ECG, echocardiography, coronary syndromes, heart failure.
- **Respiratory**: Lungs, ventilation, oxygenation, ARDS, pneumonia (as a respiratory condition), intubation/extubation, weaning, airway management, COPD, asthma, pleural disease.
- **Neurological**: Brain, spinal cord, consciousness, seizures, stroke, encephalopathy, delirium, sedation, neuromuscular disease, intracranial pressure, GCS.
- **Infectious Disease**: Sepsis, bacteremia, fungemia, antimicrobial therapy, antibiotic selection, infection source control, fever workup, cultures, resistant organisms.
- **General surgical**: Post-operative care, trauma, surgical complications, wound management, abdominal emergencies, burns, transplant perioperative care.
- **General medical**: Metabolic/endocrine disorders, renal failure, electrolyte abnormalities, acid-base disturbances, hematologic issues, GI bleeding, liver failure, nutrition, toxicology, and any ICU question not fitting the above subspecialties.

Respond ONLY with a JSON object: {"task_type": "...", "subspecialty": "..."}"""


def _load_cache() -> Dict:
    """Load classification cache from disk."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: Dict) -> None:
    """Save classification cache to disk."""
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(question: str) -> str:
    """Generate cache key from question text."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _cache_meta() -> str:
    """Cache metadata string for invalidation."""
    return f"{OLLAMA_MODEL}:{_PROMPT_VERSION}"


def _normalize_value(value: str, valid_options: List[str], fallback: str) -> str:
    """Normalize a classification value, fuzzy-matching if needed."""
    if not isinstance(value, str):
        return fallback

    value_stripped = value.strip()

    # Exact match (case-insensitive)
    for option in valid_options:
        if value_stripped.lower() == option.lower():
            return option

    # Substring match
    for option in valid_options:
        if option.lower() in value_stripped.lower() or value_stripped.lower() in option.lower():
            return option

    return fallback


def _classify_question(question: str) -> Dict[str, str]:
    """Classify a single question using Ollama. Returns dict with task_type and subspecialty."""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"Classify this ICU question:\n\n{question}\n\nRespond with JSON only: {{\"task_type\": \"...\", \"subspecialty\": \"...\"}}",
        "system": SYSTEM_PROMPT,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            response_text = result.get("response", "")
            parsed = json.loads(response_text)

            task_type = _normalize_value(
                parsed.get("task_type", ""), VALID_TASK_TYPES, "Other"
            )
            subspecialty = _normalize_value(
                parsed.get("subspecialty", ""), VALID_SUBSPECIALTIES, "General medical"
            )

            return {"task_type": task_type, "subspecialty": subspecialty}

        except (urllib.error.URLError, json.JSONDecodeError, KeyError,
                TimeoutError, OSError) as e:
            if attempt < MAX_RETRIES:
                continue
            # All retries exhausted — return fallback
            return {"task_type": "Other", "subspecialty": "General medical"}


# Module-level cache (loaded once)
_cache: Dict = {}
_cache_loaded = False


def _ensure_cache() -> Dict:
    """Lazy-load the cache."""
    global _cache, _cache_loaded
    if not _cache_loaded:
        _cache = _load_cache()
        _cache_loaded = True
    return _cache


def _get_classification(question: str) -> Dict[str, str]:
    """Get classification for a question, using cache if available."""
    cache = _ensure_cache()
    key = _cache_key(question)
    meta = _cache_meta()

    # Check cache
    if key in cache and cache[key].get("_meta") == meta:
        return {
            "task_type": cache[key]["task_type"],
            "subspecialty": cache[key]["subspecialty"]
        }

    # Classify via LLM
    result = _classify_question(question)

    # Update cache (crash-safe: write after each classification)
    cache[key] = {
        "task_type": result["task_type"],
        "subspecialty": result["subspecialty"],
        "_meta": meta,
        "_question_preview": question[:100]
    }
    _save_cache(cache)

    return result


def classify_task_type(question: str) -> str:
    """
    Classify a question into a task type using LLM.

    Args:
        question: The question text

    Returns:
        Task type: 'Diagnosis', 'Prognosis', 'Treatment', 'Knowledge', or 'Other'
    """
    return _get_classification(question)["task_type"]


def classify_subspecialty(question: str) -> str:
    """
    Classify a question into an ICU subspecialty using LLM.

    Args:
        question: The question text

    Returns:
        Subspecialty name
    """
    return _get_classification(question)["subspecialty"]


def classify_all_questions(questions: List[str]) -> Tuple[List[str], List[str]]:
    """
    Classify all questions into task types and subspecialties.

    Args:
        questions: List of question texts

    Returns:
        Tuple of (task_types, subspecialties)
    """
    task_types = []
    subspecialties = []
    for i, q in enumerate(questions):
        result = _get_classification(q)
        task_types.append(result["task_type"])
        subspecialties.append(result["subspecialty"])
        if (i + 1) % 20 == 0:
            print(f"   Classified {i + 1}/{len(questions)} questions...")
    return task_types, subspecialties


if __name__ == "__main__":
    # Test with sample ICU questions
    test_questions = [
        "What is the most likely diagnosis for a patient presenting with acute onset dyspnea, hypoxemia, and bilateral infiltrates on chest X-ray?",
        "What is the expected mortality rate for a patient with septic shock requiring three vasopressors?",
        "What antibiotic regimen should be started for suspected ventilator-associated pneumonia?",
        "Explain the pathophysiology of acute respiratory distress syndrome (ARDS).",
        "A 65-year-old patient develops ST-elevation in leads II, III, and aVF. What is the next step in management?",
    ]

    print(f"Testing LLM classifier with {len(test_questions)} questions...")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Cache: {CACHE_PATH}")
    print()

    for q in test_questions:
        result = _get_classification(q)
        print(f"Q: {q[:80]}...")
        print(f"   Task: {result['task_type']}, Subspecialty: {result['subspecialty']}")
        print()
