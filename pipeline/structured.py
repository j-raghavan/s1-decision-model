"""Generate structured-state and PowerPoint training rows with rule-computed gold labels.

These families are deliberately different from the held-out eval families in
eval/custom/generate.py (no CI triage, API retry, refunds, incident severity,
next-tool, access requests, log anomalies, ticket routing, config risk, agent
loops, slide layout, visual type, density, needs-image or split-slide). They
exercise the same skills — reading JSON state, applying policies, judging agent
traces, slide decisions — on other tasks, so the eval set stays held out.

To keep the model from learning one surface form, each row varies:
  - where the policy lives: in the question, or inside the state as a document
  - key style: snake_case or camelCase, consistently within a row
  - distractor fields that do not affect the answer
Derived quantities that need arithmetic (percent differences, durations, ages)
are precomputed into the state; the evaluation showed date arithmetic is a weak
spot, and a System One model should read such values, not compute them.

    uv run pipeline/structured.py --per-family 3000

Writes data/train/structured_v1.jsonl (git-ignored; stays on this machine).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "train" / "structured_v1.jsonl"


def camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(w.capitalize() for w in rest)


def restyle(obj, style: str):
    if style == "snake":
        return obj
    if isinstance(obj, dict):
        return {camel(k): restyle(v, style) for k, v in obj.items()}
    if isinstance(obj, list):
        return [restyle(v, style) for v in obj]
    return obj


def distractors(rng: random.Random) -> dict:
    pool = {"request_id": f"req_{rng.getrandbits(40):010x}", "region": rng.choice(["us-east-1", "eu-west-1", "ap-south-1"]),
            "schema_version": rng.choice(["1.2", "2.0", "2.1"]), "source": rng.choice(["webhook", "batch", "api"]),
            "trace_id": f"{rng.getrandbits(64):016x}", "updated_by": f"svc-{rng.choice(['ops', 'sync', 'etl'])}"}
    keys = rng.sample(list(pool), rng.randint(0, 3))
    return {k: pool[k] for k in keys}


def finish(rng: random.Random, state: dict, policy_text: str, question: dict) -> tuple[dict, dict]:
    """Place the policy in the question or the state, add distractors, restyle keys."""
    state = {**state, **distractors(rng)}
    if rng.random() < 0.5:
        state = {**state, "policy": policy_text}
        question = {**question, "instructions": question["instructions"] + " Apply the policy given in the state."}
    else:
        question = {**question, "instructions": question["instructions"] + " Policy: " + policy_text}
    return restyle(state, rng.choice(["snake", "camel"])), question


def noul(instructions: str, true: str, false: str) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}


def choice(instructions: str, criteria: dict[str, str], rng: random.Random) -> dict:
    items = list(criteria.items())
    rng.shuffle(items)
    return {"type": "choice", "instructions": instructions, "criteria": dict(items)}


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


# --------------------------------------------------------------------------
# Structured-state families
# --------------------------------------------------------------------------

def deploy_gate(rng):
    tests = rng.choice(["passed", "passed", "failed", "flaky"])
    scan = rng.choice(["clean", "clean", "low", "high", "critical"])
    approvals = rng.randint(0, 3)
    freeze = rng.random() < 0.2
    hotfix = rng.random() < 0.25
    gold = tests == "passed" and scan in ("clean", "low") and approvals >= (1 if hotfix else 2) and (not freeze or hotfix)
    state = {"release": {"version": f"v{rng.randint(1, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 9)}", "is_hotfix": hotfix},
             "checks": {"tests": tests, "security_scan_max_severity": scan, "approvals": approvals},
             "in_change_freeze_window": freeze}
    policy = ("Deploy only if tests passed, the highest security finding is clean or low, there are at least 2 approvals "
              "(1 for hotfixes), and the release is not in a change freeze window unless it is a hotfix.")
    q = noul("A release and its checks are supplied in the state. May it be deployed?", "It may be deployed.", "It must not be deployed.")
    return (*finish(rng, state, policy, q), gold)


def invoice_match(rng):
    po_amount = round(rng.uniform(200, 50000), 2)
    diff_pct = round(rng.choice([rng.uniform(0, 1.5), rng.uniform(1.5, 4), rng.uniform(4, 20)]) * rng.choice([1, -1]), 2)
    inv_amount = round(po_amount * (1 + diff_pct / 100), 2)
    qty = rng.randint(1, 500)
    received = qty if rng.random() < 0.75 else qty - rng.randint(1, max(1, qty // 3))
    vendor_ok = rng.random() < 0.85
    gold = abs(diff_pct) <= 2 and received == qty and vendor_ok
    state = {"purchase_order": {"id": f"PO-{rng.randint(10000, 99999)}", "vendor_id": "V-118", "amount": po_amount, "quantity": qty},
             "invoice": {"vendor_id": "V-118" if vendor_ok else "V-221", "amount": inv_amount},
             "goods_receipt": {"quantity_received": received},
             "amount_difference_pct": diff_pct}
    policy = "Approve payment only when vendors match, the invoice is within 2% of the PO amount, and all ordered units were received."
    q = noul("A purchase order, invoice and goods receipt are supplied in the state. Should the invoice be approved for payment?",
             "Approve the invoice.", "Hold the invoice.")
    return (*finish(rng, state, policy, q), gold)


def sla_breach(rng):
    priority = rng.choice(["P1", "P2", "P3", "P4"])
    limits = {"P1": 15, "P2": 60, "P3": 240, "P4": 1440}
    minutes = rng.choice([rng.randint(1, 20), rng.randint(20, 90), rng.randint(90, 600), rng.randint(600, 3000)])
    paused = rng.randint(0, minutes // 2) if rng.random() < 0.3 else 0
    gold = minutes - paused > limits[priority]
    state = {"ticket": {"id": f"INC-{rng.randint(1000, 9999)}", "priority": priority},
             "minutes_to_first_response": minutes, "minutes_paused_awaiting_customer": paused,
             "sla_first_response_minutes": limits}
    policy = "A breach occurs when minutes to first response, minus minutes paused awaiting the customer, exceed the limit for the ticket's priority."
    q = noul("A support ticket's response timing is supplied in the state. Was the first-response SLA breached?",
             "The SLA was breached.", "The SLA was met.")
    return (*finish(rng, state, policy, q), gold)


def shipment_exception(rng):
    kinds = {"address_issue": "Delivery attempted: address not found", "customs_hold": "Held by customs: documentation required",
             "damaged": "Package damaged in transit; claim opened", "lost": "No scan for 10 days; trace initiated",
             "weather_delay": "Delayed: severe weather at hub", "on_track": "Out for delivery"}
    gold = rng.choice(list(kinds))
    events = [{"status": "Label created"}, {"status": "Picked up"}, {"status": "Arrived at sort facility"}]
    events = events[: rng.randint(1, 3)] + [{"status": kinds[gold]}]
    state = {"shipment": {"tracking": f"1Z{rng.getrandbits(40):010X}", "carrier": rng.choice(["UPS", "DHL", "FedEx"])}, "events": events}
    q = choice("A shipment's tracking events are supplied in the state. What is its current exception status (latest event)?",
               {"address_issue": "Address problem", "customs_hold": "Held at customs", "damaged": "Damaged", "lost": "Lost or missing",
                "weather_delay": "Weather delay", "on_track": "No exception; on track"}, rng)
    return restyle({**state, **distractors(rng)}, rng.choice(["snake", "camel"])), q, gold


def tool_error_kind(rng):
    results = {"rate_limited": {"status": 429, "error": "Too Many Requests", "retry_after_s": rng.choice([1, 5, 30])},
               "auth_error": {"status": 401, "error": "invalid or expired token"},
               "not_found": {"status": 404, "error": f"resource {rng.randint(100, 999)} not found"},
               "invalid_arguments": {"status": 400, "error": "field 'limit' must be <= 100", "field": "limit"},
               "timeout": {"status": None, "error": "upstream timed out after 30000 ms"},
               "success": {"status": 200, "data": {"items": rng.randint(0, 40)}}}
    gold = rng.choice(list(results))
    state = {"agent_step": rng.randint(1, 12), "tool_call": {"name": rng.choice(["list_orders", "get_user", "search_docs"]),
                                                             "arguments": {"limit": rng.choice([10, 50, 500])}},
             "tool_result": results[gold]}
    q = choice("An agent's latest tool call and its result are supplied in the state. What happened?",
               {"rate_limited": "Rate limited", "auth_error": "Authentication failed", "not_found": "Resource not found",
                "invalid_arguments": "Invalid arguments", "timeout": "Timed out", "success": "Succeeded"}, rng)
    return restyle({**state, **distractors(rng)}, rng.choice(["snake", "camel"])), q, gold


def agent_escalate(rng):
    confidence = round(rng.uniform(0.2, 0.99), 2)
    attempts = rng.randint(1, 5)
    sentiment = rng.choice(["positive", "neutral", "frustrated", "angry"])
    asked_human = rng.random() < 0.15
    gold = asked_human or confidence < 0.5 or attempts >= 3 or sentiment == "angry"
    state = {"conversation": {"turns": rng.randint(2, 30), "user_sentiment": sentiment, "user_asked_for_human": asked_human},
             "agent": {"answer_confidence": confidence, "failed_resolution_attempts": attempts}}
    policy = ("Escalate to a human if the user asked for one, the agent's confidence is below 0.5, there have been 3 or more "
              "failed resolution attempts, or the user is angry.")
    q = noul("A support conversation and the agent's status are supplied in the state. Should the agent escalate to a human?",
             "Escalate to a human.", "The agent should continue.")
    return (*finish(rng, state, policy, q), gold)


PEOPLE = ["Priya", "Marcus", "Aiko", "Lena", "Tomás", "Grace", "Omar", "Ines", "Kwame", "Sofia"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
CUSTOMERS = ["ACME", "Globex", "Initech", "Umbrella", "Stark Industries", "Wayne Enterprises", "Hooli", "Vandelay"]
SERVICES = ["billing", "search", "checkout", "auth", "notifications", "reporting"]


def agent_goal_done(rng):
    person, day, hour = rng.choice(PEOPLE), rng.choice(DAYS), rng.randint(9, 17)
    cust, svc, team = rng.choice(CUSTOMERS), rng.choice(SERVICES), rng.choice(["finance", "legal", "ops", "sales"])
    inv = f"INV-{rng.randint(1000, 9999)}"
    templates = [
        (f"Create a calendar event 'Design review' for {day} {hour}:00 and invite {person}.",
         [("create_event", {"id": f"evt_{rng.randint(10, 99)}", "start": f"{day} {hour}:00"}),
          ("invite", {"invited": [f"{person.lower()}@example.com"]})]),
        (f"Find the latest invoice for {cust} and email it to {team}.",
         [("search_invoices", {"customer": cust, "latest": inv}),
          ("send_email", {"to": f"{team}@example.com", "attachment": f"{inv}.pdf", "sent": True})]),
        (f"Rotate the API key for the {svc} service, update the secret store, and restart the service.",
         [("rotate_key", {"service": svc, "new_key_id": f"k_{rng.randint(10, 99)}"}),
          ("put_secret", {"name": f"{svc}/api_key", "version": rng.randint(2, 20)}),
          ("restart_service", {"service": svc, "status": "running"})]),
        (f"Export {cust}'s Q{rng.randint(1, 4)} usage report and share it with {person}.",
         [("export_report", {"customer": cust, "file": f"usage_{cust.lower().replace(' ', '_')}.csv"}),
          ("share_file", {"shared_with": [f"{person.lower()}@example.com"]})]),
    ]
    goal, steps = rng.choice(templates)
    n = rng.randint(1, len(steps))
    history = [{"call": c, "result": dict(r)} for c, r in steps[:n]]
    failed = rng.random() < 0.25
    if failed:
        history[rng.randrange(n)]["result"] = {"error": rng.choice(["permission denied", "timeout", "invalid argument"])}
    gold = n == len(steps) and not failed
    q = noul("An agent's goal and tool-call history are supplied in the state. Has the goal been fully completed, with every "
             "required step done and no step failing?", "The goal is complete.", "The goal is not complete.")
    return restyle({"goal": goal, "history": history}, rng.choice(["snake", "camel"])), q, gold


def payload_validation(rng):
    name = rng.choice(PEOPLE).lower().replace("á", "a")
    payload = {"email": f"{name}.{rng.randint(1, 999)}@{rng.choice(['example.com', 'corp.io', 'mail.net'])}",
               "age": rng.randint(1, 120), "country_code": rng.choice(["US", "DE", "IN", "BR", "JP", "FR", "NG", "CA", "MX", "KE"]),
               "start_date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"}
    bad = rng.choice(["email", "age", "country_code", "start_date", "none"])
    if bad == "email":
        user, domain = payload["email"].split("@")
        payload["email"] = rng.choice([f"{user}.{domain}", f"{user}@", f"@{domain}", f"{user}@@{domain}", f"{user} @{domain}"])
    elif bad == "age":
        payload["age"] = rng.choice([rng.randint(-50, 0), rng.randint(121, 400), str(rng.randint(18, 90)) + " years", None])
    elif bad == "country_code":
        payload["country_code"] = rng.choice([rng.choice(["USA", "DEU", "IND"]), payload["country_code"].lower(),
                                              rng.choice(["Germany", "Brazil", "Japan"]), f"{rng.randint(10, 99)}", ""])
    elif bad == "start_date":
        d = payload["start_date"]
        payload["start_date"] = rng.choice([f"2026-{rng.randint(13, 19)}-{d[-2:]}", f"{d[5:7]}/{d[-2:]}/2026",
                                            f"2026-{d[5:7]}-{rng.randint(32, 39)}", rng.choice(["tomorrow", "next week", "TBD"])])
    state = {"payload": payload, "schema": {"email": "valid email address", "age": "integer from 1 to 120",
                                            "country_code": "ISO 3166-1 alpha-2, two uppercase letters", "start_date": "YYYY-MM-DD"}}
    q = choice("A request payload and its schema are supplied in the state. Which field is invalid?",
               {"email": "email", "age": "age", "country_code": "country_code", "start_date": "start_date",
                "none": "None; the payload is valid"}, rng)
    return restyle(state, rng.choice(["snake", "camel"])) if rng.random() < 0.3 else state, q, bad


def pr_risk(rng):
    sensitive = rng.random() < 0.35
    lines = rng.choice([rng.randint(1, 50), rng.randint(50, 400), rng.randint(400, 3000)])
    tests_added = rng.random() < 0.6
    migration = rng.random() < 0.15
    level = 0
    if lines > 50 or not tests_added:
        level = 1
    if (sensitive and not tests_added) or lines > 400 or migration:
        level = 2
    if sensitive and (migration or lines > 400):
        level = 3
    paths = ["src/ui/button.tsx", "docs/README.md", "src/api/orders.py"]
    if sensitive:
        paths.append(rng.choice(["src/auth/session.py", "src/payments/charge.py"]))
    if migration:
        paths.append("migrations/0042_add_index.sql")
    state = {"pull_request": {"files_changed": paths, "lines_changed": lines, "tests_added": tests_added,
                              "touches_auth_or_payments": sensitive, "has_db_migration": migration}}
    policy = ("Low: small (50 lines or fewer) with tests. Medium: larger than 50 lines or no tests. High: auth/payments code without "
              "tests, more than 400 lines, or a database migration. Critical: auth/payments code together with a migration or more "
              "than 400 lines.")
    q = score("A pull request is supplied in the state. Rate its review risk.", ["low", "medium", "high", "critical"])
    return (*finish(rng, state, policy, q), level)


def alert_same_incident(rng):
    same_service = rng.random() < 0.6
    gap = rng.choice([rng.randint(0, 10), rng.randint(10, 60), rng.randint(60, 600)])
    same_sig = rng.random() < 0.6
    gold = same_service and same_sig and gap <= 15
    a = {"service": "checkout", "signature": "DB_CONN_TIMEOUT", "fired_at_minute": 0}
    b = {"service": "checkout" if same_service else rng.choice(["search", "auth"]),
         "signature": "DB_CONN_TIMEOUT" if same_sig else rng.choice(["HTTP_5XX_SPIKE", "DISK_PRESSURE"]), "fired_at_minute": gap}
    state = {"alert_a": a, "alert_b": b, "minutes_between_alerts": gap}
    policy = "Two alerts belong to the same incident when they share the service and the error signature and fired within 15 minutes."
    q = noul("Two alerts are supplied in the state. Do they belong to the same incident?", "Same incident.", "Separate incidents.")
    return (*finish(rng, state, policy, q), gold)


def kyc_check(rng):
    age = rng.choice([rng.randint(14, 17), rng.randint(18, 80)])
    doc = rng.random() < 0.8
    sanctions = rng.random() < 0.08
    address = rng.random() < 0.85
    gold = age >= 18 and doc and not sanctions and address
    state = {"applicant": {"id": f"A{rng.randint(10000, 99999)}", "age_years": age, "id_document_verified": doc,
                           "sanctions_list_match": sanctions, "address_verified": address}}
    policy = "Pass KYC only for adults (18+) with a verified ID document and address and no sanctions-list match."
    q = noul("A customer's verification record is supplied in the state. Does the customer pass KYC?", "Passes KYC.", "Fails KYC.")
    return (*finish(rng, state, policy, q), gold)


def churn_risk(rng):
    drop = rng.choice([rng.randint(-20, 10), rng.randint(10, 40), rng.randint(40, 90)])
    tickets = rng.randint(0, 12)
    days_left = rng.choice([rng.randint(0, 30), rng.randint(30, 120), rng.randint(120, 365)])
    level = 0
    if drop > 10 or tickets > 3:
        level = 1
    if drop > 40 or (tickets > 6 and days_left <= 90):
        level = 2
    if drop > 40 and days_left <= 30:
        level = 3
    state = {"account": {"plan": rng.choice(["team", "business", "enterprise"]), "usage_drop_pct_90d": drop,
                         "support_tickets_90d": tickets, "days_until_renewal": days_left}}
    policy = ("Low by default. Medium if usage fell more than 10% or there were more than 3 tickets. High if usage fell more than 40%, "
              "or more than 6 tickets with renewal within 90 days. Critical if usage fell more than 40% with renewal within 30 days.")
    q = score("A customer account is supplied in the state. Rate its churn risk.", ["low", "medium", "high", "critical"])
    return (*finish(rng, state, policy, q), level)


# --------------------------------------------------------------------------
# PowerPoint families
# --------------------------------------------------------------------------

METRICS = ["revenue", "signups", "weekly active users", "churn", "support tickets", "gross margin", "deal size", "page load time",
           "NPS", "conversion rate", "ad spend", "units sold"]
GROUPS = ["region", "product line", "channel", "customer segment", "sales rep", "plan tier"]
PERIODS = ["since January", "over the last 12 months", "quarter by quarter", "week over week", "since launch"]


def ppt_chart_for_question(rng):
    m, m2, g, p = rng.choice(METRICS), rng.choice(METRICS), rng.choice(GROUPS), rng.choice(PERIODS)
    asks = {"line": [f"How has {m} changed {p}?", f"What is the trend in {m} {p}?", f"Is {m} going up or down {p}?"],
            "bar": [f"Which {g} has the highest {m}?", f"How does {m} compare across each {g}?", f"Rank each {g} by {m}."],
            "stacked_bar": [f"How is {m} split by {g} in each quarter?", f"What share of each month's {m} comes from each {g}?"],
            "histogram": [f"How is {m} distributed across customers?", f"What is the spread of {m}?", f"How many accounts fall in each {m} band?"],
            "scatter": [f"Is {m} related to {m2}?", f"Do accounts with higher {m} also have higher {m2}?"]}
    gold = rng.choice(list(asks))
    q = choice("The question a slide must answer is supplied in the state. Which chart answers it best?",
               {"line": "Line chart (trend over time)", "bar": "Bar chart (compare categories)",
                "stacked_bar": "Stacked bar (composition per group)", "histogram": "Histogram (distribution)",
                "scatter": "Scatter plot (relationship between two measures)"}, rng)
    return {"slide": {"audience": rng.choice(["board", "team", "customers", "investors"]), "question": rng.choice(asks[gold])}}, q, gold


def ppt_title_quality(rng):
    m, g = rng.choice(METRICS), rng.choice(GROUPS)
    pct, n = rng.randint(3, 80), rng.randint(2, 12)
    level = rng.choice([0, 1, 2])
    if level == 2:
        title = rng.choice([f"{m.capitalize()} rose {pct}% after the {rng.choice(['redesign', 'price change', 'launch'])}",
                            f"Top {g} drove {pct}% of {m} growth", f"Adding {n} engineers cuts {m} by {pct}%",
                            f"{m.capitalize()} fell {pct}% in Q{rng.randint(1, 4)}"])
    elif level == 1:
        title = rng.choice([f"{m.capitalize()} by {g}", f"{m.capitalize()} trends this quarter", f"Review of {m} across each {g}"])
    else:
        title = rng.choice([m.capitalize(), "Overview", "Update", f"Some thoughts on {m}",
                            f"An overview of the many different factors that may have influenced {m} across every {g} this year"])
    state = {"slide": {"title": title, "title_word_count": len(title.split()), "title_has_number": bool(re.search(r"\d", title))}}
    q = score("A slide title is supplied in the state. Rate it. Weak: a bare topic word or vague phrase, or longer than 12 words. "
              "Adequate: a clear topic without a takeaway. Strong: states the takeaway, usually with a number or a result.",
              ["weak", "adequate", "strong"])
    return state, q, level


def ppt_speaker_notes(rng):
    audience = rng.choice(["board", "executives", "team", "all_hands"])
    has_chart = rng.random() < 0.5
    numbers = rng.randint(0, 12)
    gold = (has_chart and audience in ("board", "executives")) or numbers > 6
    state = {"slide": {"audience": audience, "has_chart": has_chart, "numeric_values_on_slide": numbers,
                       "title": rng.choice(["Q3 results", "Pipeline health", "Hiring update", "Roadmap"])}}
    policy = "Speaker notes are required for any chart shown to the board or executives, and for any slide with more than 6 numbers."
    q = noul("A slide is supplied in the state. Does it need speaker notes?", "It needs speaker notes.", "Speaker notes are optional.")
    return (*finish(rng, state, policy, q), gold)


FAMILIES = {
    "train_deploy_gate": deploy_gate, "train_invoice_match": invoice_match, "train_sla_breach": sla_breach,
    "train_shipment_exception": shipment_exception, "train_tool_error_kind": tool_error_kind, "train_agent_escalate": agent_escalate,
    "train_agent_goal_done": agent_goal_done, "train_payload_validation": payload_validation, "train_pr_risk": pr_risk,
    "train_alert_same_incident": alert_same_incident, "train_kyc_check": kyc_check, "train_churn_risk": churn_risk,
    "train_ppt_chart_for_question": ppt_chart_for_question, "train_ppt_title_quality": ppt_title_quality,
    "train_ppt_speaker_notes": ppt_speaker_notes,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-family", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261005)
    args = ap.parse_args()

    rows = []
    for fam, gen in FAMILIES.items():
        rng = random.Random(f"{args.seed}:{fam}")
        golds: list = []
        seen: set[str] = set()
        attempts = 0
        while len(golds) < args.per_family:
            attempts += 1
            if attempts > 400 * args.per_family:
                print(f"{fam}: stopped at {len(golds)} unique balanced rows (generator variety exhausted)")
                break
            state, q, gold = gen(rng)
            n_options = 2 if q["type"] == "noul" else len(q["criteria"])
            if golds.count(gold) >= -(-args.per_family // n_options):
                continue
            key = json.dumps([state, q["instructions"]], sort_keys=True)
            if key in seen:  # exact duplicates add nothing
                continue
            seen.add(key)
            rows.append({"row_id": f"{fam}-{len(golds)}", "family": fam, "task_type": q["type"], "state": state, "question": q,
                         "gold": gold, "n_options": n_options,
                         "provenance": {"source": "rule generator", "generator": "pipeline/structured.py", "license": "Apache-2.0",
                                        "origin": "this repo; no model involved"}})
            golds.append(gold)
        counts = collections.Counter(map(str, golds))
        if len(set(counts.values())) > 1:
            # Variety ran out unevenly: downsample every answer to the rarest one's count so the family stays balanced.
            floor, kept, fam_rows = min(counts.values()), collections.Counter(), [r for r in rows if r["family"] == fam]
            rows = [r for r in rows if r["family"] != fam]
            for r in fam_rows:
                if kept[str(r["gold"])] < floor:
                    kept[str(r["gold"])] += 1
                    rows.append(r)
            golds = [r["gold"] for r in rows if r["family"] == fam]
        dist = dict(sorted(collections.Counter(map(str, golds)).items()))
        print(f"{fam:30s} {q['type']:6s} {len(golds):5d} rows  {dist}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total {len(rows)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
