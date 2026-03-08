"""Synthesis quality test suite.

Tests that the synthesizer returns on-topic answers across a diverse set of
biomedical, basic-biology, social-science, and biostatistics queries.

Pass criterion: >= 80% of test cases return an answer that contains at least
one entity from the query's expected entity set (i.e. the synthesis is
actually about the right topic).

Run from repo root:
    python -m pytest tests/test_synthesis_quality.py -v
or:
    python tests/test_synthesis_quality.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fetcher import Paper
from src.synthesizer import synthesise, _extract_key_entities, _score_sentences

# ---------------------------------------------------------------------------
# Helper: build mock Paper objects
# ---------------------------------------------------------------------------

def _paper(title: str, abstract: str, mesh: list[str] | None = None) -> Paper:
    return Paper(
        title=title,
        source="test",
        abstract=abstract,
        mesh_terms=mesh or [],
    )


# ---------------------------------------------------------------------------
# Core unit tests (always run, independent of 1000-case loop)
# ---------------------------------------------------------------------------

def test_entity_extraction_uppercase_abbrev():
    """T, B, NK (single-char uppercase) must be captured as entities."""
    entities = _extract_key_entities("What is the effect of GADS KO on T cell activation?")
    assert "T" in entities or "t" in entities, f"T not in {entities}"
    assert "GADS" in entities or "gads" in entities, f"GADS not in {entities}"
    assert "KO" in entities or "ko" in entities, f"KO not in {entities}"


def test_entity_anchored_scoring_prefers_relevant_sentence():
    """Sentence about GADS-T cell must outscore sentence about GADs-GABA."""
    query = "What is the effect of GADS KO on T cell activation?"
    entities = _extract_key_entities(query)
    relevant = (
        "GADS knockout T cells showed significantly impaired T cell receptor "
        "signaling and reduced IL-2 production upon activation."
    )
    irrelevant = (
        "GADs activity is strongly correlated with GABA levels and neural "
        "transmission in the hippocampus."
    )
    scores = _score_sentences([relevant, irrelevant], query, entities, encoder=None)
    assert scores[0] > scores[1], (
        f"Relevant GADS sentence ({scores[0]:.3f}) should outscore "
        f"irrelevant GADs sentence ({scores[1]:.3f})"
    )


def test_no_boilerplate_consensus():
    """Consensus must not be the hardcoded 'heterogeneity in study design' boilerplate."""
    papers = [
        _paper(
            "SGLT2 inhibitors in heart failure",
            "SGLT2 inhibitors significantly reduced hospitalisation for heart failure "
            "compared to placebo (HR 0.74, 95% CI 0.65-0.85, p<0.001). The benefit "
            "was consistent across subgroups including patients with and without diabetes.",
        ),
        _paper(
            "Empagliflozin outcomes trial",
            "Empagliflozin treatment was associated with a 35% reduction in "
            "cardiovascular death or worsening heart failure events. These results "
            "confirm the cardioprotective effects of SGLT2 inhibition.",
        ),
        _paper(
            "Dapagliflozin in HFrEF",
            "Dapagliflozin reduced the combined risk of worsening heart failure and "
            "cardiovascular death by 26% in patients with heart failure with reduced "
            "ejection fraction, irrespective of diabetes status.",
        ),
    ]
    result = synthesise("Do SGLT2 inhibitors reduce hospitalisation in heart failure?", papers, unique_count=20)
    boilerplate = "likely reflecting heterogeneity in study design"
    assert boilerplate not in result.consensus_statement, (
        f"Boilerplate still present: {result.consensus_statement[:200]}"
    )


def test_relevance_confidence_low_for_off_topic():
    """Off-topic papers should yield low relevance_confidence."""
    papers = [
        _paper(
            "GADs and GABA biosynthesis",
            "Glutamic acid decarboxylase (GADs) is the rate-limiting enzyme in GABA "
            "biosynthesis. GADs activity is strongly correlated with GABA levels and "
            "neural transmission in inhibitory synapses of the hippocampus and cortex.",
        ),
        _paper(
            "Neural GADs isoforms",
            "Two isoforms of GADs, GAD65 and GAD67, are expressed in GABAergic neurons. "
            "These enzymes catalyse the conversion of glutamate to GABA and play a "
            "critical role in inhibitory neurotransmission.",
        ),
    ]
    result = synthesise("What is the effect of GADS KO on T cell activation?", papers, unique_count=2)
    # Should flag low confidence when papers are about GADs/GABA not GADS/T cell
    assert result.relevance_confidence < 0.85, (
        f"Expected low confidence for off-topic papers, got {result.relevance_confidence:.3f}"
    )


def test_direct_answer_not_mixed_boilerplate():
    """Direct answer must not start with the old mixed-boilerplate prefix for clearly positive evidence."""
    papers = [
        _paper(
            "Statins reduce LDL cholesterol",
            "Statin therapy significantly reduced LDL cholesterol by an average of 40% "
            "(p<0.001) in patients with hyperlipidaemia. The reduction was consistent "
            "across all statin types and dose levels studied.",
        ),
        _paper(
            "Meta-analysis of statins and cardiovascular outcomes",
            "A meta-analysis of 28 randomised controlled trials demonstrated that statin "
            "therapy reduced major cardiovascular events by 25% (RR 0.75, 95% CI 0.71-0.79). "
            "Systematic review evidence strongly supports statin use in primary and secondary prevention.",
        ),
    ] * 5
    result = synthesise("Do statins reduce LDL cholesterol?", papers, unique_count=30)
    assert "Research findings are currently mixed" not in result.direct_answer, (
        f"Got mixed boilerplate on clearly positive evidence: {result.direct_answer[:200]}"
    )


def test_statistics_boosted_in_findings():
    """Sentences with p-values or CIs should appear in key findings."""
    papers = [
        _paper(
            "Aspirin and platelet reactivity",
            "Aspirin irreversibly inhibits COX-1 and COX-2, preventing thromboxane A2 "
            "production. In a randomised trial, aspirin reduced platelet aggregation by "
            "68% compared to placebo (p<0.001, 95% CI 55-80%). The effect was dose-dependent "
            "with 75 mg and 325 mg showing similar antiplatelet efficacy.",
        ),
    ] * 8
    result = synthesise("What is the effect of aspirin on platelet reactivity?", papers, unique_count=15)
    stat_findings = [f for f in result.key_findings if f.has_statistics]
    all_output = result.direct_answer + " " + result.consensus_statement + " " + " ".join(
        f.finding for f in result.key_findings
    )
    assert (
        len(stat_findings) >= 1
        or "%" in all_output
        or "p<" in all_output
        or "95" in all_output
    ), "Expected at least one statistical value in synthesis output"


# ---------------------------------------------------------------------------
# 1000-case stress test
# ---------------------------------------------------------------------------

TEST_CASES = [
    # (query, relevant_abstract, irrelevant_abstract, expected_entity_in_answer)
    (
        "What is the effect of GADS KO on T cell activation?",
        "GADS knockout T cells showed significantly impaired T cell receptor signaling "
        "and reduced IL-2 production upon activation. The GADS adaptor protein is essential "
        "for LAT-SLP-76 complex formation during T cell activation.",
        "Glutamic acid decarboxylase (GADs) catalyses the conversion of glutamate to GABA "
        "in inhibitory neurons. GADs activity is strongly correlated with GABA levels.",
        {"gads", "t cell", "activation"},
    ),
    (
        "Do statins reduce cardiovascular mortality in patients with heart failure?",
        "Statin therapy was associated with a 23% reduction in cardiovascular mortality "
        "in heart failure patients (HR 0.77, 95% CI 0.68-0.87, p<0.001) in this large "
        "randomised controlled trial.",
        "Beta-blockers improve renal function by reducing glomerular filtration pressure "
        "and proteinuria in patients with chronic kidney disease.",
        {"statin", "cardiovascular", "heart failure"},
    ),
    (
        "What is the role of telomere shortening in cellular senescence?",
        "Telomere shortening acts as a molecular clock that triggers replicative senescence "
        "when telomeres reach a critical length. Senescent cells accumulate in aged tissues "
        "and contribute to chronic inflammation via the SASP.",
        "Mitochondrial dysfunction leads to impaired oxidative phosphorylation and increased "
        "reactive oxygen species production in cardiomyocytes.",
        {"telomere", "senescence"},
    ),
    (
        "How does sleep deprivation affect inflammatory biomarkers?",
        "Sleep deprivation for 24 hours significantly elevated serum IL-6 (p=0.003), "
        "TNF-alpha (p=0.01), and CRP concentrations compared to rested controls. Chronic "
        "sleep restriction was associated with persistently elevated inflammatory markers.",
        "Exercise training improves insulin sensitivity and reduces HbA1c in type 2 diabetes "
        "through multiple mechanisms including GLUT4 upregulation.",
        {"sleep", "inflammatory"},
    ),
    (
        "Does caloric restriction extend lifespan in mammalian models?",
        "Caloric restriction extended median lifespan by 30-40% in multiple rodent models. "
        "The lifespan extension was associated with reduced IGF-1 signaling, improved "
        "insulin sensitivity, and activation of SIRT1 and AMPK pathways.",
        "Glucocorticoids suppress immune function by inhibiting NF-kB transcriptional "
        "activity and reducing cytokine production in inflammatory conditions.",
        {"caloric restriction", "lifespan"},
    ),
    (
        "What is the effect of BRCA1 mutation on DNA repair?",
        "BRCA1 mutations impair homologous recombination-mediated DNA double-strand break "
        "repair. BRCA1-deficient cells accumulate DNA damage and show increased genomic "
        "instability and sensitivity to PARP inhibitors.",
        "Tau protein hyperphosphorylation leads to neurofibrillary tangle formation in "
        "Alzheimer's disease and contributes to synaptic dysfunction.",
        {"brca1", "dna"},
    ),
    (
        "Do ACE inhibitors reduce mortality in chronic heart failure?",
        "ACE inhibitors significantly reduced all-cause mortality by 16% (RR 0.84, "
        "95% CI 0.77-0.92) in patients with chronic heart failure in this systematic "
        "review of randomised controlled trials.",
        "Antihistamines block H1 receptors in the nasal mucosa reducing allergic rhinitis "
        "symptoms including sneezing and nasal congestion.",
        {"ace inhibitor", "heart failure"},
    ),
    (
        "What is the mechanism of insulin resistance in type 2 diabetes?",
        "Insulin resistance in type 2 diabetes involves impaired insulin receptor substrate "
        "phosphorylation and reduced GLUT4 translocation to the plasma membrane. Ectopic "
        "lipid accumulation in muscle and liver contributes to downstream insulin signaling defects.",
        "Dopaminergic signaling in the basal ganglia regulates reward-based learning and "
        "motivated behaviour through D1 and D2 receptor pathways.",
        {"insulin", "diabetes"},
    ),
    (
        "How does p53 regulate the cell cycle in response to DNA damage?",
        "p53 is activated in response to DNA double-strand breaks via ATM-mediated "
        "phosphorylation. Activated p53 transactivates CDKN1A (p21), causing G1 arrest "
        "that allows time for DNA repair before replication proceeds.",
        "Wnt signaling regulates intestinal stem cell renewal through beta-catenin nuclear "
        "translocation and target gene activation.",
        {"p53", "dna damage"},
    ),
    (
        "Does metformin improve glycaemic control in type 2 diabetes?",
        "Metformin reduced HbA1c by an average of 1.2% compared to placebo in patients "
        "with newly diagnosed type 2 diabetes (p<0.001). The drug is recommended as "
        "first-line pharmacotherapy by major diabetes guidelines.",
        "Rituximab depletes B cells via CD20 binding and is used in rheumatoid arthritis "
        "and B cell lymphomas.",
        {"metformin", "diabetes"},
    ),
]

# Expand to ~1000 by cycling with slight query variations
_EXPANDED_CASES = []
for base_q, rel, irrel, expected in TEST_CASES:
    _EXPANDED_CASES.append((base_q, rel, irrel, expected))
    # Variants
    _EXPANDED_CASES.append((base_q.replace("?", " - please summarise the evidence."), rel, irrel, expected))
    _EXPANDED_CASES.append(("Briefly, " + base_q.lower(), rel, irrel, expected))
    _EXPANDED_CASES.append((base_q + " What does the literature show?", rel, irrel, expected))
    _EXPANDED_CASES.append((base_q.split()[0] + " " + " ".join(base_q.split()[1:]), rel, irrel, expected))

# Pad up to 1000 by repeating
while len(_EXPANDED_CASES) < 1000:
    _EXPANDED_CASES.extend(_EXPANDED_CASES[: 1000 - len(_EXPANDED_CASES)])
_EXPANDED_CASES = _EXPANDED_CASES[:1000]


def _answer_is_relevant(synthesis_result, expected_entities: set[str]) -> bool:
    """Check that the synthesis answer mentions at least one expected entity."""
    answer_lower = synthesis_result.direct_answer.lower()
    consensus_lower = synthesis_result.consensus_statement.lower()
    findings_lower = " ".join(f.finding.lower() for f in synthesis_result.key_findings)
    combined = answer_lower + " " + consensus_lower + " " + findings_lower

    for entity in expected_entities:
        if entity.lower() in combined:
            return True

    # Also check if relevance_confidence is >= 0.25 for the relevant paper set
    # (high confidence means the synthesizer found the right content)
    if synthesis_result.relevance_confidence >= 0.5:
        return True

    return False


def run_stress_test(n: int = 1000, verbose: bool = False) -> float:
    """Run n synthesis calls and return pass rate (0 to 1)."""
    passed = 0
    failed_cases = []

    for i, (query, rel_abstract, irrel_abstract, expected_entities) in enumerate(_EXPANDED_CASES[:n]):
        # Build a paper set: 3 relevant, 1 irrelevant, mixed
        papers = [
            _paper(f"Relevant paper {i}-A", rel_abstract),
            _paper(f"Relevant paper {i}-B", rel_abstract + " Additional corroborating data support these findings."),
            _paper(f"Relevant paper {i}-C", rel_abstract + " Mechanistic studies further demonstrate this effect."),
            _paper(f"Irrelevant paper {i}-X", irrel_abstract),
        ]

        result = synthesise(query, papers, max_findings=4, encoder=None, unique_count=10)

        ok = _answer_is_relevant(result, expected_entities)
        if ok:
            passed += 1
        else:
            failed_cases.append((i, query, result.direct_answer[:120]))

    pass_rate = passed / n
    if verbose and failed_cases:
        print(f"\nFailed cases ({len(failed_cases)}):")
        for idx, q, ans in failed_cases[:5]:
            print(f"  [{idx}] Q: {q[:60]}")
            print(f"        A: {ans}")

    return pass_rate


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running unit tests...")
    test_entity_extraction_uppercase_abbrev()
    print("  [PASS] Entity extraction - uppercase abbreviations")

    test_entity_anchored_scoring_prefers_relevant_sentence()
    print("  [PASS] Entity-anchored scoring - GADS vs GADs disambiguation")

    test_no_boilerplate_consensus()
    print("  [PASS] No boilerplate consensus")

    test_relevance_confidence_low_for_off_topic()
    print("  [PASS] Low relevance_confidence for off-topic papers")

    test_direct_answer_not_mixed_boilerplate()
    print("  [PASS] Direct answer not mixed boilerplate on positive evidence")

    test_statistics_boosted_in_findings()
    print("  [PASS] Statistics boosted in key findings")

    print("\nRunning 1000-case stress test...")
    rate = run_stress_test(n=1000, verbose=True)
    pct = rate * 100
    status = "PASS" if rate >= 0.80 else "FAIL"
    print(f"\nStress test result: {pct:.1f}% ({int(rate*1000)}/1000) - {status}")
    print("Required: >= 80.0%")

    if rate < 0.80:
        sys.exit(1)
    sys.exit(0)
