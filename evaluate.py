import json
import os
import random
import requests
from sentence_transformers import SentenceTransformer, util

API_URL = "http://localhost:8000/query"
EVAL_FILE = os.path.join(os.path.dirname(__file__), "data", "eval_incidents", "eval_dataset.json")
SEMANTIC_THRESHOLD = 0.7

_model = SentenceTransformer("all-MiniLM-L6-v2")
_embedding_cache: dict[str, any] = {}


def normalize(text: str) -> str:
    return text.lower().strip().replace("-", " ")


def _get_embedding(text: str):
    if text not in _embedding_cache:
        _embedding_cache[text] = _model.encode(text, convert_to_tensor=True)
    return _embedding_cache[text]


def semantic_match(predicted: str, ground_truth: str) -> bool:
    p = normalize(predicted)
    g = normalize(ground_truth)
    if not p or not g:
        return False
    similarity = util.cos_sim(_get_embedding(p), _get_embedding(g)).item()
    return similarity >= SEMANTIC_THRESHOLD


_QUERY_TEMPLATES = [
    "Facing issue with {}",
    "Seeing {} in production",
    "System showing {} after deployment",
    "Users reporting {}",
    "Why am I getting {}?",
    "Experiencing {} in service",
]


def generate_query(symptoms: list[str]) -> str:
    if not symptoms:
        return ""
    symptom = random.choice(symptoms[:3])
    template = random.choice(_QUERY_TEMPLATES)
    return template.format(symptom)


def call_api(query: str) -> dict | None:
    try:
        resp = requests.post(API_URL, json={"query": query}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [API error] {e}")
        return None


def load_incidents(eval_file: str) -> list[dict]:
    try:
        with open(eval_file) as f:
            records = json.load(f)
    except Exception as e:
        print(f"  [error] Could not load {eval_file}: {e}")
        return []

    incidents = []
    for record in records:
        if not record.get("symptoms") or not record.get("confirmed_root_cause"):
            print(f"  [skip] {record.get('incident_id', '?')}: missing required fields")
            continue
        incidents.append(record)
    return incidents


def check_top_k_retrieval(incident: dict, similar_incidents: list[dict]) -> tuple[bool, bool]:
    incident_id = incident.get("incident_id", "")
    ground_truth = normalize(incident["confirmed_root_cause"])

    top1_match = False
    top3_match = False

    for i, sim in enumerate(similar_incidents[:3]):
        sim_id = sim.get("incident_id", sim.get("id", ""))
        sim_cause = normalize(sim.get("confirmed_root_cause", sim.get("root_cause", "")))

        matched = (sim_id == incident_id) or semantic_match(sim_cause, ground_truth)
        if matched:
            if i == 0:
                top1_match = True
            top3_match = True
            break

    return top1_match, top3_match


def evaluate():
    print(f"Loading incidents from: {EVAL_FILE}")
    incidents = load_incidents(EVAL_FILE)
    if not incidents:
        print("No valid incidents found. Exiting.")
        return

    print("Pre-computing ground truth embeddings...")
    for incident in incidents:
        _get_embedding(normalize(incident["confirmed_root_cause"]))

    total = 0
    rc_correct = 0
    top1_correct = 0
    top3_correct = 0
    hard_cases = 0
    hard_correct = 0
    conf_correct = []
    conf_incorrect = []

    for incident in incidents:
        iid = incident.get("incident_id", "?")
        query = generate_query(incident["symptoms"])
        ground_truth = incident["confirmed_root_cause"]

        print(f"  Evaluating {iid}...", end=" ", flush=True)
        response = call_api(query)
        if response is None:
            print("skipped (API error)")
            continue

        total += 1
        predicted = response.get("likely_root_cause", "")
        confidence = response.get("confidence", 0.0)
        similar = response.get("similar_incidents", [])

        rc_match = semantic_match(predicted, ground_truth)
        top1, top3 = check_top_k_retrieval(incident, similar)

        query_words = set(normalize(query).split())
        truth_words = set(normalize(ground_truth).split())
        keyword_overlap = len(query_words & truth_words) / max(len(truth_words), 1)
        is_hard = keyword_overlap < 0.2
        if is_hard:
            hard_cases += 1
            if rc_match:
                hard_correct += 1

        if rc_match:
            rc_correct += 1
            conf_correct.append(confidence)
        else:
            conf_incorrect.append(confidence)

        if top1:
            top1_correct += 1
        if top3:
            top3_correct += 1

        status = "OK" if rc_match else "MISS"
        hard_tag = " [hard]" if is_hard else ""
        print(f"[{status}]{hard_tag} conf={confidence:.2f}")

    if total == 0:
        print("\nNo samples evaluated.")
        return

    avg_conf_correct = sum(conf_correct) / len(conf_correct) if conf_correct else 0.0
    avg_conf_incorrect = sum(conf_incorrect) / len(conf_incorrect) if conf_incorrect else 0.0

    hard_acc = hard_correct / hard_cases * 100 if hard_cases else 0.0

    print()
    print("=" * 45)
    print(f"Total samples:              {total}")
    print(f"Root Cause Accuracy:        {rc_correct / total * 100:.1f}%")
    print(f"Top-1 Retrieval Accuracy:   {top1_correct / total * 100:.1f}%")
    print(f"Top-3 Retrieval Accuracy:   {top3_correct / total * 100:.1f}%")
    print(f"Hard-case Accuracy:         {hard_acc:.1f}%  ({hard_correct}/{hard_cases})")
    print()
    print(f"Avg Confidence (correct):   {avg_conf_correct:.2f}")
    print(f"Avg Confidence (incorrect): {avg_conf_incorrect:.2f}")
    print("=" * 45)


if __name__ == "__main__":
    evaluate()
