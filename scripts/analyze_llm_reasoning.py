"""
Analyze LLM reasoning to understand directional bias and lack of "unchanged" predictions.
"""

import pickle


def analyze_llm_reasoning():
    """Analyze the reasoning patterns in LLM predictions."""

    print("🔍 ANALYZING LLM REASONING PATTERNS")
    print("=" * 60)

    # Load LLM predictions
    pred_file = "/Users/chiraag/Projects/gwu/lab/apriomics/output/qualitative_predictions_no_context_gpt_4o_mini_2024_07_18_temp01_53metabolites.pkl"

    try:
        with open(pred_file, "rb") as f:
            llm_predictions = pickle.load(f)
    except FileNotFoundError:
        print("❌ Prediction file not found!")
        return

    print(f"📁 Using: {pred_file.split('/')[-1]}")
    print(f"🔢 Total predictions: {len(llm_predictions)}")

    # Categorize predictions
    increases = []
    decreases = []
    unchanged = []

    for metabolite, pred in llm_predictions.items():
        direction = pred.get("prediction", "unknown").lower()
        reasoning = pred.get("reasoning", "")
        magnitude = pred.get("magnitude", "unknown")
        confidence = pred.get("confidence", 0.0)

        entry = {
            "metabolite": metabolite,
            "direction": direction,
            "magnitude": magnitude,
            "confidence": confidence,
            "reasoning": reasoning,
            "reasoning_length": len(reasoning),
        }

        if direction == "increase":
            increases.append(entry)
        elif direction == "decrease":
            decreases.append(entry)
        else:
            unchanged.append(entry)

    print("\n📊 DIRECTION DISTRIBUTION:")
    print("=" * 60)
    print(
        f"Increases: {len(increases)} ({len(increases) / len(llm_predictions) * 100:.1f}%)"
    )
    print(
        f"Decreases: {len(decreases)} ({len(decreases) / len(llm_predictions) * 100:.1f}%)"
    )
    print(
        f"Unchanged: {len(unchanged)} ({len(unchanged) / len(llm_predictions) * 100:.1f}%)"
    )

    # Analyze reasoning patterns for increases
    print("\n📈 INCREASE REASONING PATTERNS:")
    print("=" * 60)

    increase_keywords = []
    for entry in increases[:10]:  # First 10 increases
        reasoning = entry["reasoning"].lower()
        print(f"\n{entry['metabolite']} (conf={entry['confidence']:.2f}):")
        print(
            f"   {entry['reasoning'][:200]}{'...' if len(entry['reasoning']) > 200 else ''}"
        )

        # Extract key phrases
        diabetes_phrases = [
            "diabetes",
            "diabetic",
            "insulin resistance",
            "glucose",
            "hyperglycemia",
            "metabolic dysfunction",
            "oxidative stress",
            "inflammation",
        ]

        found_phrases = [phrase for phrase in diabetes_phrases if phrase in reasoning]
        if found_phrases:
            increase_keywords.extend(found_phrases)

    # Analyze reasoning patterns for decreases
    print("\n📉 DECREASE REASONING PATTERNS:")
    print("=" * 60)

    decrease_keywords = []
    for entry in decreases:  # All decreases since there are fewer
        reasoning = entry["reasoning"].lower()
        print(f"\n{entry['metabolite']} (conf={entry['confidence']:.2f}):")
        print(
            f"   {entry['reasoning'][:200]}{'...' if len(entry['reasoning']) > 200 else ''}"
        )

        # Extract key phrases
        decrease_phrases = [
            "decreased",
            "reduced",
            "lower",
            "depleted",
            "clearance",
            "excretion",
            "kidney function",
            "renal",
            "filtration",
        ]

        found_phrases = [phrase for phrase in decrease_phrases if phrase in reasoning]
        if found_phrases:
            decrease_keywords.extend(found_phrases)

    # Analyze why no "unchanged" predictions
    print("\n❓ WHY NO 'UNCHANGED' PREDICTIONS?")
    print("=" * 60)

    if len(unchanged) == 0:
        print("🤔 HYPOTHESIS: LLM may be biased to predict change because:")
        print("   1. Diabetes is a 'disease state' - LLM expects everything to change")
        print("   2. We asked about 'Type 2 diabetes vs healthy' - implies differences")
        print("   3. LLM trained on literature that emphasizes significant findings")
        print("   4. No explicit guidance about normal/unchanged metabolites")
    else:
        print("Found some unchanged predictions - analyzing...")
        for entry in unchanged:
            print(f"\n{entry['metabolite']}:")
            print(
                f"   {entry['reasoning'][:200]}{'...' if len(entry['reasoning']) > 200 else ''}"
            )

    # Common reasoning patterns
    print("\n🔑 COMMON REASONING PATTERNS:")
    print("=" * 60)

    all_reasoning = " ".join(
        [pred.get("reasoning", "").lower() for pred in llm_predictions.values()]
    )

    # Count key medical terms
    medical_terms = {
        "diabetes": all_reasoning.count("diabetes"),
        "insulin": all_reasoning.count("insulin"),
        "glucose": all_reasoning.count("glucose"),
        "oxidative stress": all_reasoning.count("oxidative stress"),
        "inflammation": all_reasoning.count("inflammation"),
        "kidney": all_reasoning.count("kidney"),
        "renal": all_reasoning.count("renal"),
        "metabolism": all_reasoning.count("metabolism"),
        "increased": all_reasoning.count("increased"),
        "decreased": all_reasoning.count("decreased"),
        "unchanged": all_reasoning.count("unchanged"),
        "normal": all_reasoning.count("normal"),
    }

    print("Medical term frequencies:")
    for term, count in sorted(medical_terms.items(), key=lambda x: x[1], reverse=True):
        print(f"   {term}: {count}")

    # Confidence analysis
    print("\n🎯 CONFIDENCE ANALYSIS:")
    print("=" * 60)

    increase_conf = [e["confidence"] for e in increases]
    decrease_conf = [e["confidence"] for e in decreases]

    print(
        f"Increase confidence: μ={sum(increase_conf) / len(increase_conf):.3f}, range=[{min(increase_conf):.2f}, {max(increase_conf):.2f}]"
    )
    print(
        f"Decrease confidence: μ={sum(decrease_conf) / len(decrease_conf):.3f}, range=[{min(decrease_conf):.2f}, {max(decrease_conf):.2f}]"
    )

    # Key insights
    print("\n💡 KEY INSIGHTS:")
    print("=" * 60)

    bias_ratio = len(increases) / len(decreases) if len(decreases) > 0 else float("inf")
    print(
        f"1. Increase/Decrease ratio: {bias_ratio:.1f}:1 (should be closer to 2:1 or 3:1)"
    )

    if "increased" in medical_terms and "decreased" in medical_terms:
        word_bias = medical_terms["increased"] / medical_terms["decreased"]
        print(f"2. 'Increased' vs 'Decreased' language bias: {word_bias:.1f}:1")

    if medical_terms.get("unchanged", 0) == 0:
        print("3. Zero mentions of 'unchanged' - LLM may not consider this possibility")

    avg_reasoning_length = sum(
        len(pred.get("reasoning", "")) for pred in llm_predictions.values()
    ) / len(llm_predictions)
    print(f"4. Average reasoning length: {avg_reasoning_length:.0f} characters")

    print("\n🎯 RECOMMENDATIONS:")
    print("=" * 60)
    print("1. Add explicit 'unchanged' examples in prompt")
    print("2. Emphasize that many metabolites may be unchanged in diabetes")
    print("3. Ask LLM to consider baseline/normal levels explicitly")
    print("4. Include counter-examples (metabolites that shouldn't change)")
    print("5. Reframe question to be less 'disease-centric'")


if __name__ == "__main__":
    analyze_llm_reasoning()
