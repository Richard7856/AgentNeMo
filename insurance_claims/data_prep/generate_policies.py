"""
generate_policies.py  -  creates three realistic sample insurance policy PDFs.

WHY generate synthetic PDFs instead of downloading real ones:
- Real policy PDFs have complex layouts (tables, headers, footers) that trip up
  naive PDF parsers and would obscure what we're learning about RAG
- Synthetic content lets us control exactly what the RAG should and shouldn't know
- We can write specific coverage amounts and exclusions that we'll test in eval queries
- The content is modeled on HDI Seguros / standard P&C policy language

Run once: uv run python insurance_claims/data_prep/generate_policies.py
Output: data/policies/auto_policy_sample.pdf, home_policy_sample.pdf, life_policy_sample.pdf
"""

from fpdf import FPDF

# ============================================================================
# Helper
# ============================================================================

def create_pdf(title: str, sections: list[tuple[str, str]]) -> FPDF:
    """Creates a simple two-column-section PDF with title and content blocks."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, title, ln=True, align="C")
    pdf.ln(4)

    # Divider
    pdf.set_draw_color(100, 100, 100)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, heading, ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, body)
        pdf.ln(3)

    return pdf


# ============================================================================
# Policy content
# ============================================================================

AUTO_SECTIONS = [
    (
        "Policy Overview",
        (
            "Policy Number: AUTO-2025-001234\n"
            "Policy Holder: [Insured Name]\n"
            "Policy Period: January 1, 2025 to December 31, 2025\n"
            "Insurer: Sample Insurance Company S.A. de C.V.\n"
            "Vehicle: 2022 Volkswagen Jetta, VIN: WVWZZZ16ZAM123456\n"
            "Sum Insured: MXN $420,000 (agreed value)"
        ),
    ),
    (
        "Coverage A  -  Collision and Total Loss",
        (
            "Covers damage to the insured vehicle resulting from collision with another vehicle, "
            "object, or rollover. In the event of total loss (repair cost exceeds 75% of the "
            "agreed value), the insurer will pay the full agreed value of MXN $420,000 minus "
            "the applicable deductible.\n\n"
            "Deductible: 3% of agreed value (MXN $12,600) per occurrence.\n"
            "Coverage limit: Agreed value MXN $420,000."
        ),
    ),
    (
        "Coverage B  -  Third-Party Liability",
        (
            "Covers bodily injury and property damage caused to third parties as a result of "
            "the operation of the insured vehicle.\n\n"
            "Bodily Injury limit: MXN $3,000,000 per occurrence.\n"
            "Property Damage limit: MXN $600,000 per occurrence.\n"
            "Legal defense included up to MXN $250,000.\n"
            "Coverage applies within Mexico and the United States border zone (100km)."
        ),
    ),
    (
        "Coverage C  -  Medical Payments (Occupants)",
        (
            "Covers reasonable and necessary medical expenses for the driver and up to 4 "
            "passengers injured in a covered accident, regardless of fault.\n\n"
            "Limit: MXN $100,000 per person.\n"
            "Includes: hospital expenses, ambulance, surgery, rehabilitation up to 12 months."
        ),
    ),
    (
        "Coverage D  -  Theft and Vandalism",
        (
            "Total theft: pays agreed value MXN $420,000 minus 10% deductible (MXN $42,000).\n"
            "Partial theft: covers stolen components (wheels, radio, catalytic converter) "
            "up to MXN $50,000 per occurrence with 5% deductible.\n"
            "Vandalism: covers intentional damage by third parties up to MXN $30,000 "
            "with mandatory police report filed within 24 hours."
        ),
    ),
    (
        "Coverage E  -  Roadside Assistance",
        (
            "24/7 roadside assistance included at no additional cost:\n"
            "- Towing: up to 50 km from breakdown location\n"
            "- Tire change and battery jump-start\n"
            "- Fuel delivery (cost of fuel charged to insured)\n"
            "- Vehicle lockout service\n"
            "- Hotel accommodation up to MXN $2,000/night if breakdown occurs >100km from home"
        ),
    ),
    (
        "Exclusions",
        (
            "This policy does NOT cover:\n"
            "1. Damage caused while the vehicle is being operated by an unlicensed driver.\n"
            "2. Damage caused while the driver has a blood alcohol level above 0.08%.\n"
            "3. Mechanical breakdown, wear and tear, or tire blowouts unrelated to an accident.\n"
            "4. Intentional damage caused by the insured.\n"
            "5. Racing or speed tests on any track or road.\n"
            "6. Damage occurring outside of Mexico unless Coverage B is explicitly extended.\n"
            "7. Loss due to nuclear, biological, or chemical events."
        ),
    ),
    (
        "Claims Procedure",
        (
            "To file a claim:\n"
            "1. Report the incident within 72 hours of occurrence.\n"
            "2. Do not admit liability or sign any document without insurer authorization.\n"
            "3. File a police report for theft, vandalism, or third-party incidents.\n"
            "4. Submit: completed claim form, police report (if applicable), photos of damage, "
            "driver license of all parties, vehicle registration, repair estimate.\n"
            "5. Claims are adjudicated within 5 business days of receiving complete documentation.\n"
            "6. Payment issued within 10 business days of claim approval."
        ),
    ),
    (
        "Premium and Payment",
        (
            "Annual premium: MXN $28,500 (including 16% VAT)\n"
            "Payment options: annual, semi-annual (+5% surcharge), or monthly (+12% surcharge)\n"
            "Grace period: 30 days from due date before policy lapses.\n"
            "Late payment: policy suspended after 30-day grace period. Reinstatement requires "
            "payment of all outstanding premiums plus a 2% reinstatement fee."
        ),
    ),
]

HOME_SECTIONS = [
    (
        "Policy Overview",
        (
            "Policy Number: HOME-2025-005678\n"
            "Policy Holder: [Insured Name]\n"
            "Policy Period: March 1, 2025 to February 28, 2026\n"
            "Insurer: Sample Insurance Company S.A. de C.V.\n"
            "Property Address: [Insured Property Address]\n"
            "Construction Type: Masonry (Type A)\n"
            "Sum Insured  -  Building: MXN $2,500,000\n"
            "Sum Insured  -  Contents: MXN $500,000"
        ),
    ),
    (
        "Coverage A  -  Fire and Allied Perils",
        (
            "Covers physical damage to the insured building and its contents caused by:\n"
            "- Fire, lightning, explosion\n"
            "- Smoke damage from a sudden and accidental event\n"
            "- Aircraft or vehicle impact\n"
            "- Riots, strikes, civil commotion\n\n"
            "Limit: MXN $2,500,000 for building, MXN $500,000 for contents.\n"
            "Deductible: 2% of insured value (minimum MXN $5,000)."
        ),
    ),
    (
        "Coverage B  -  Water Damage",
        (
            "Covers accidental and sudden water damage from:\n"
            "- Burst or leaking pipes, plumbing, or water heaters\n"
            "- Overflow of fixed appliances (washing machines, dishwashers)\n"
            "- Rain entering through broken windows or doors caused by storm\n\n"
            "NOT covered: gradual leaks, seepage, groundwater flooding.\n"
            "Limit: MXN $300,000 per occurrence.\n"
            "Deductible: MXN $8,000 per occurrence."
        ),
    ),
    (
        "Coverage C  -  Theft and Burglary",
        (
            "Covers theft of contents following forced entry (burglary) into the insured premises.\n\n"
            "Limit: MXN $200,000 per occurrence (sublimit for jewelry/electronics: MXN $50,000).\n"
            "Deductible: 5% of claim amount (minimum MXN $3,000).\n"
            "Requirements: Police report mandatory within 48 hours of discovery.\n"
            "High-value items (>MXN $10,000 each) must be scheduled on the policy to be covered.\n"
            "NOT covered: theft without forced entry, theft by household employees, shoplifting."
        ),
    ),
    (
        "Coverage D  -  Liability",
        (
            "Personal liability protection for bodily injury or property damage caused to third "
            "parties on or around the insured premises.\n\n"
            "Limit: MXN $1,500,000 per occurrence.\n"
            "Covers: slip-and-fall accidents, dog bites, falling objects.\n"
            "Includes legal defense costs up to MXN $100,000.\n"
            "NOT covered: intentional acts, business activities conducted on premises, "
            "liability arising from vehicles."
        ),
    ),
    (
        "Coverage E  -  Additional Living Expenses",
        (
            "If a covered loss makes the insured premises uninhabitable, covers:\n"
            "- Hotel or temporary rental accommodation: up to MXN $20,000/month\n"
            "- Additional meal expenses above normal baseline: up to MXN $8,000/month\n"
            "- Pet boarding: up to MXN $3,000/month\n\n"
            "Maximum duration: 12 months or until repairs are complete, whichever is sooner.\n"
            "Total limit for this coverage: MXN $350,000."
        ),
    ),
    (
        "Exclusions",
        (
            "This policy does NOT cover:\n"
            "1. Earthquake, volcanic eruption, tsunami (separate earthquake endorsement available).\n"
            "2. Flood from external water sources (rivers, ocean, storm surge).\n"
            "3. Gradual deterioration, wear and tear, or lack of maintenance.\n"
            "4. Mold or fungal damage unless caused by a covered water damage event.\n"
            "5. War, terrorism, or government seizure.\n"
            "6. Business inventory or equipment stored at the insured premises.\n"
            "7. Damage to land, trees, plants, or lawns (Coverage A only covers the structure).\n"
            "8. Damage caused by the insured, household members, or invited guests intentionally."
        ),
    ),
    (
        "Claims Procedure",
        (
            "To file a claim:\n"
            "1. Notify insurer within 48 hours of discovering the damage.\n"
            "2. Take reasonable steps to prevent further damage (e.g., cover broken windows).\n"
            "3. Do NOT discard damaged items before adjuster inspection.\n"
            "4. File police report for theft or vandalism.\n"
            "5. Submit: completed claim form, photos, inventory of damaged items with values, "
            "repair estimates, proof of ownership for high-value items.\n"
            "6. Adjuster assigned within 3 business days of notice.\n"
            "7. Settlement issued within 15 business days of complete documentation."
        ),
    ),
]

LIFE_SECTIONS = [
    (
        "Policy Overview",
        (
            "Policy Number: LIFE-2025-009012\n"
            "Policy Holder / Insured: [Insured Name]\n"
            "Policy Type: Whole Life with Savings Component\n"
            "Policy Period: Lifetime (coverage until age 99)\n"
            "Issue Date: June 1, 2025\n"
            "Insurer: Sample Insurance Company S.A. de C.V.\n"
            "Death Benefit: MXN $5,000,000\n"
            "Primary Beneficiary: [Name]  -  70%\n"
            "Secondary Beneficiary: [Name]  -  30%"
        ),
    ),
    (
        "Coverage A  -  Death Benefit",
        (
            "Upon the death of the insured from any cause (subject to exclusions below), "
            "the insurer will pay the death benefit of MXN $5,000,000 to the named beneficiary(ies).\n\n"
            "Payment is made within 20 business days of receiving complete claim documentation.\n"
            "The death benefit is exempt from income tax under Mexican fiscal law.\n"
            "No deductible applies to the death benefit."
        ),
    ),
    (
        "Coverage B  -  Accidental Death Benefit (Double Indemnity)",
        (
            "If death results from an accident (external, sudden, violent, and involuntary cause), "
            "an additional accidental death benefit of MXN $5,000,000 is paid, for a total "
            "death benefit of MXN $10,000,000.\n\n"
            "Conditions:\n"
            "- Death must occur within 180 days of the accident.\n"
            "- Accident must be the direct and sole cause of death.\n"
            "- Police or coroner report required confirming accidental cause."
        ),
    ),
    (
        "Coverage C  -  Total and Permanent Disability (TPD)",
        (
            "If the insured becomes totally and permanently disabled before age 65, "
            "the insurer will:\n"
            "1. Waive all future premium payments.\n"
            "2. Pay MXN $2,500,000 (50% of death benefit) as a lump-sum disability benefit.\n\n"
            "Definition of TPD: inability to perform any occupation for which the insured "
            "is qualified by education, training, or experience, for at least 24 consecutive months.\n"
            "Two independent medical examinations required to confirm TPD."
        ),
    ),
    (
        "Coverage D  -  Critical Illness Accelerated Benefit",
        (
            "Upon first diagnosis of a covered critical illness, the insured may accelerate "
            "up to 50% of the death benefit (MXN $2,500,000).\n\n"
            "Covered critical illnesses: cancer (stages III-IV), heart attack, stroke, "
            "kidney failure requiring dialysis, organ transplant, blindness, paralysis.\n\n"
            "Waiting period: 90 days from policy issue date.\n"
            "The amount accelerated is deducted from the death benefit payable at death."
        ),
    ),
    (
        "Savings and Cash Value",
        (
            "This policy accumulates a savings reserve (cash value) over time.\n\n"
            "Guaranteed interest rate: 3.5% per annum on the savings component.\n"
            "After year 3: insured may request a policy loan of up to 90% of cash value.\n"
            "After year 5: insured may surrender the policy for its full cash value.\n\n"
            "Surrender values (guaranteed minimums):\n"
            "Year 5: MXN $120,000\n"
            "Year 10: MXN $380,000\n"
            "Year 20: MXN $1,100,000\n"
            "Year 30: MXN $2,300,000"
        ),
    ),
    (
        "Exclusions",
        (
            "The death benefit will NOT be paid if death results from:\n"
            "1. Suicide within the first 2 years of policy issue (return of premiums paid only).\n"
            "2. War, armed conflict, or participation in military operations.\n"
            "3. Intentional self-inflicted injury (other than suicide after year 2).\n"
            "4. Commission of a felony by the insured.\n"
            "5. Drug or alcohol intoxication.\n"
            "6. Extreme sports or aviation as pilot (unless specifically endorsed).\n\n"
            "Double indemnity exclusions (accidental death):\n"
            "- Death caused by illness or disease (even if accident contributed).\n"
            "- Death occurring more than 180 days after the accident."
        ),
    ),
    (
        "Claims Procedure",
        (
            "To file a death benefit claim:\n"
            "1. Notify the insurer within 30 days of the insured's death.\n"
            "2. Submit: completed beneficiary claim form, original death certificate (notarized), "
            "government-issued ID of all beneficiaries, policy document.\n"
            "3. For accidental death: add police report, coroner/autopsy report.\n"
            "4. For TPD: two independent medical reports + insurer's medical examination.\n"
            "5. For critical illness acceleration: specialist diagnosis report + pathology results.\n\n"
            "Payment timeline: 20 business days from receipt of complete documentation.\n"
            "The insurer may request additional medical records up to 60 days post-claim."
        ),
    ),
    (
        "Premium Structure",
        (
            "Monthly premium: MXN $8,400 (fixed for life of policy)\n"
            "Breakdown: MXN $5,200 risk coverage + MXN $3,200 savings component\n\n"
            "Grace period: 31 days after premium due date.\n"
            "Policy lapse: premiums unpaid after grace period. Reinstatable within 2 years "
            "with evidence of insurability and payment of overdue premiums plus interest.\n"
            "Premium waiver: automatically triggered by TPD (Coverage C)."
        ),
    ),
]


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Generates all three sample policy PDFs into data/policies/."""
    import os

    output_dir = "data/policies"
    os.makedirs(output_dir, exist_ok=True)

    policies = [
        ("Sample Auto Insurance Policy", AUTO_SECTIONS, "auto_policy_sample.pdf"),
        ("Sample Homeowners Insurance Policy", HOME_SECTIONS, "home_policy_sample.pdf"),
        ("Sample Whole Life Insurance Policy", LIFE_SECTIONS, "life_policy_sample.pdf"),
    ]

    for title, sections, filename in policies:
        pdf = create_pdf(title, sections)
        path = os.path.join(output_dir, filename)
        pdf.output(path)
        print(f"Generated: {path}")

    print("\nAll policies generated. Run ingest_policies.py next to build the FAISS index.")


if __name__ == "__main__":
    main()
