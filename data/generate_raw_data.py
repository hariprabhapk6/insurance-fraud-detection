"""
generate_raw_data.py
=====================
Generates a synthetic but realistic RAW insurance-claims dataset.

Why synthetic? This environment has no internet access, so a public
Kaggle-style fraud dataset cannot be downloaded. Instead we generate data
with a known ground-truth fraud-generating process (so every downstream
model can be honestly evaluated) BUT we deliberately inject the kind of
mess real raw data has: missing values, duplicate rows, inconsistent
string casing/categories, mixed date formats, outliers, and free-text
descriptions. This raw file is what 01_ml/preprocessing.py cleans.

Outputs:
  data/raw/claims_raw.csv            - main tabular claims (messy)
  data/raw/claim_events_raw.csv      - sequential per-claim event log (for DL/RNN)
  data/raw/policyholder_network.csv  - edges between policyholders who share
                                        agents/addresses/phone numbers (for GNN)
"""
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
random.seed(42)

N_CLAIMS = 6000
N_POLICYHOLDERS = 2200

CITIES = ["Chennai", "Coimbatore", "Mumbai", "Delhi", "Bengaluru", "Hyderabad",
          "Pune", "Kolkata", "Ahmedabad", "Jaipur", None]
CITY_DIRTY_VARIANTS = {
    "Chennai": ["Chennai", "chennai", "CHENNAI", "Chennai "],
    "Coimbatore": ["Coimbatore", "coimbatore", "Cbe", "COIMBATORE"],
    "Mumbai": ["Mumbai", "mumbai", "Bombay", "MUMBAI"],
    "Delhi": ["Delhi", "delhi", "New Delhi", "DELHI"],
    "Bengaluru": ["Bengaluru", "bengaluru", "Bangalore", "BLR"],
}
CLAIM_TYPES = ["Auto", "Health", "Property", "Life", "Travel"]
CLAIM_TYPE_DIRTY = {
    "Auto": ["Auto", "auto", "AUTO", "Vehicle"],
    "Health": ["Health", "health", "HEALTH", "Medical"],
    "Property": ["Property", "property", "Home"],
    "Life": ["Life", "life", "LIFE"],
    "Travel": ["Travel", "travel", "TRAVEL"],
}
GENDER = ["M", "F", "Other"]
OCCUPATIONS = ["Salaried", "Self-Employed", "Business Owner", "Retired", "Student", "Unemployed"]
AGENTS = [f"AGT{100+i}" for i in range(40)]

TEXT_TEMPLATES_LEGIT = [
    "Vehicle was rear-ended at a traffic signal on {city} main road, minor bumper damage, police report filed.",
    "Patient admitted for scheduled surgery, all documents and hospital bills attached as per policy.",
    "Kitchen fire caused smoke damage to walls and ceiling, fire department report attached.",
    "Flight was delayed by 9 hours due to weather, claiming for meal and hotel expenses as per policy terms.",
    "Policyholder passed away due to natural causes, death certificate and medical history submitted.",
    "Minor collision in parking lot, both parties exchanged insurance details, repair estimate attached.",
    "Routine dental procedure covered under health plan, invoice and prescription enclosed.",
    "Heavy rains caused water damage to the ground floor, photos and plumber assessment attached.",
]
TEXT_TEMPLATES_SUSPICIOUS = [
    "Vehicle stolen from an unspecified location, no witnesses, no police FIR number provided yet.",
    "Claim filed immediately after policy purchase for a pre-existing condition not disclosed earlier.",
    "Multiple claims filed within the same week for unrelated incidents, documentation inconsistent.",
    "Total loss claim for vehicle with no photos, repair shop cannot be verified, contact number invalid.",
    "Hospital named on the bill does not match records, treatment dates overlap with another claim.",
    "Property damage claim filed right before policy renewal, inflated repair estimate from unlisted vendor.",
    "Same bank account used across three different claimant names in the past month.",
    "Injury claim with no medical report, claimant requests direct cash settlement urgently.",
]

def dirty_city(city):
    if city is None:
        return None
    variants = CITY_DIRTY_VARIANTS.get(city, [city])
    return random.choice(variants)

def dirty_claim_type(ct):
    variants = CLAIM_TYPE_DIRTY.get(ct, [ct])
    return random.choice(variants)

def random_date(start, end):
    delta = end - start
    d = start + timedelta(days=random.randint(0, delta.days), hours=random.randint(0,23))
    # inject mixed date formats as strings (raw-data messiness)
    fmt = random.choice(["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%b-%Y"])
    return d, d.strftime(fmt)

def main():
    start = datetime(2022, 1, 1)
    end = datetime(2025, 12, 31)

    # ---- Policyholders ----
    policyholders = [f"PH{10000+i}" for i in range(N_POLICYHOLDERS)]
    ph_agent = {ph: random.choice(AGENTS) for ph in policyholders}
    ph_phone = {ph: f"9{RNG.integers(100000000,999999999)}" for ph in policyholders}
    ph_address_cluster = {ph: RNG.integers(0, 300) for ph in policyholders}  # shared-address groups

    # A small ring of policyholders sharing the same agent+bank account -> fraud ring
    fraud_ring = random.sample(policyholders, 25)
    ring_bank_account = "ACC-RING-9911"
    ph_bank = {ph: (ring_bank_account if ph in fraud_ring else f"ACC-{RNG.integers(100000,999999)}")
               for ph in policyholders}

    # ---- Claims (tabular) ----
    rows = []
    for i in range(N_CLAIMS):
        claim_id = f"CLM{500000+i}"
        ph = random.choice(policyholders)
        ctype = random.choice(CLAIM_TYPES)
        dt_obj, dt_str = random_date(start, end)

        age = int(np.clip(RNG.normal(42, 14), 18, 85))
        income = float(np.clip(RNG.normal(650000, 250000), 80000, 3000000))
        policy_tenure_months = int(RNG.integers(1, 180))
        claim_amount = float(np.clip(RNG.lognormal(mean=10.5, sigma=0.9), 2000, 3000000))
        prior_claims = int(RNG.poisson(0.6))
        is_ring_member = ph in fraud_ring

        # ground-truth fraud generating process (hidden signal, mimics real risk factors)
        fraud_score = (
            0.22 * prior_claims
            + (2.6 if is_ring_member else 0)
            + (1.3 if policy_tenure_months < 3 else 0)
            + (1.1 if claim_amount > 500000 else 0)
            + (0.7 if ctype == "Auto" and claim_amount > 300000 else 0)
            + (0.6 if income and income < 200000 and claim_amount > 400000 else 0)
            + RNG.normal(0, 0.75)
        )
        is_fraud = 1 if fraud_score > 1.35 else 0

        desc = random.choice(TEXT_TEMPLATES_SUSPICIOUS if is_fraud and random.random() < 0.7
                              else TEXT_TEMPLATES_LEGIT)
        city_choice = random.choice(CITIES)
        desc = desc.format(city=city_choice if city_choice else "the")

        row = {
            "claim_id": claim_id,
            "policyholder_id": ph,
            "claim_type": dirty_claim_type(ctype),
            "claim_date": dt_str,
            "claimant_age": age if random.random() > 0.02 else None,   # missing values
            "annual_income": round(income, 2) if random.random() > 0.03 else None,
            "policy_tenure_months": policy_tenure_months,
            "claim_amount": round(claim_amount, 2),
            "prior_claims_count": prior_claims,
            "city": dirty_city(city_choice),
            "gender": random.choice(GENDER) if random.random() > 0.01 else None,
            "occupation": random.choice(OCCUPATIONS),
            "agent_id": ph_agent[ph],
            "bank_account": ph_bank[ph],
            "claim_description": desc,
            "is_fraud": is_fraud,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # inject exact + near duplicates
    dupes = df.sample(120, random_state=1).copy()
    df = pd.concat([df, dupes], ignore_index=True)

    # inject a few extreme outliers
    outlier_idx = df.sample(15, random_state=2).index
    df.loc[outlier_idx, "claim_amount"] = df.loc[outlier_idx, "claim_amount"] * 50

    # inject stray whitespace / inconsistent NA markers
    na_markers = ["NA", "N/A", "?", "", "unknown", None]
    for col in ["occupation", "gender"]:
        idx = df.sample(frac=0.02, random_state=3).index
        df.loc[idx, col] = [random.choice(na_markers) for _ in idx]

    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)  # shuffle
    df.to_csv("/home/claude/fraud_project/data/raw/claims_raw.csv", index=False)

    # ---- Sequential claim events (for DL/RNN) ----
    event_types = ["SUBMITTED", "DOC_REQUESTED", "DOC_RECEIVED", "ADJUSTER_ASSIGNED",
                   "SITE_VISIT", "AMOUNT_REVISED", "ESCALATED", "APPROVED", "REJECTED", "PAID"]
    event_rows = []
    for _, r in df.drop_duplicates(subset="claim_id").iterrows():
        n_events = RNG.integers(3, 8)
        if r["is_fraud"] == 1:
            # fraudulent claims: skip adjuster assignment more often, escalate,
            # get amount revised, and are more often rejected
            weights = np.array([3, 2, 0.5, 0.5, 3, 3, 1, 2, 0.5])
        else:
            # legitimate claims: routine, mostly approved/paid path
            weights = np.array([1, 2, 3, 1.5, 0.5, 0.3, 3, 0.3, 2.5])
        probs = weights / weights.sum()
        seq = ["SUBMITTED"] + list(RNG.choice(event_types[1:], size=n_events - 1, p=probs))
        base_date = datetime(2023, 1, 1)
        for step, ev in enumerate(seq):
            event_rows.append({
                "claim_id": r["claim_id"],
                "step": step,
                "event_type": ev,
                "event_time": (base_date + timedelta(days=step * random.randint(1, 5))).isoformat(),
                "amount_at_step": round(r["claim_amount"] * RNG.uniform(0.85, 1.15), 2),
            })
    events_df = pd.DataFrame(event_rows)
    events_df.to_csv("/home/claude/fraud_project/data/raw/claim_events_raw.csv", index=False)

    # ---- Policyholder network (for GNN-style features) ----
    edges = []
    for ph in policyholders:
        # edge if sharing agent or address cluster with another random policyholder
        other = random.choice(policyholders)
        if other != ph:
            shared_agent = ph_agent[ph] == ph_agent[other]
            shared_addr = ph_address_cluster[ph] == ph_address_cluster[other]
            shared_bank = ph_bank[ph] == ph_bank[other]
            if shared_agent or shared_addr or shared_bank:
                edges.append({"source": ph, "target": other,
                               "shared_agent": shared_agent,
                               "shared_address_cluster": shared_addr,
                               "shared_bank_account": shared_bank})
    # ensure the fraud ring is densely connected
    for a in fraud_ring:
        for b in random.sample(fraud_ring, 5):
            if a != b:
                edges.append({"source": a, "target": b, "shared_agent": False,
                               "shared_address_cluster": False, "shared_bank_account": True})
    edges_df = pd.DataFrame(edges).drop_duplicates()
    edges_df.to_csv("/home/claude/fraud_project/data/raw/policyholder_network.csv", index=False)

    print(f"claims_raw.csv:            {df.shape}")
    print(f"claim_events_raw.csv:      {events_df.shape}")
    print(f"policyholder_network.csv:  {edges_df.shape}")
    print(f"Fraud rate (raw, pre-clean): {df['is_fraud'].mean():.3f}")

if __name__ == "__main__":
    main()
