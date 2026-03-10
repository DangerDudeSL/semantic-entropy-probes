"""
Unit tests for sentence_segmenter and claim_filter modules.

Run with:
    cd d:\\Github Repositories\\semantic-entropy-probes
    python -m pytest test_sentence_segmenter.py -v

No GPU or model required — tests the NLP segmentation layer only.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from sentence_segmenter import split_sentences, split_sentences_with_spans, get_backend_name
from claim_filter import is_claim, filter_claims


# ═══════════════════════════════════════════════════════════════
#  SENTENCE SEGMENTER TESTS
# ═══════════════════════════════════════════════════════════════

class TestSentenceSegmenter:
    """Tests for split_sentences and split_sentences_with_spans."""

    def test_backend_detection(self):
        """Verify that a backend is detected."""
        name = get_backend_name()
        assert name in ("spacy", "pysbd", "regex"), f"Unknown backend: {name}"
        print(f"  [INFO] Using backend: {name}")

    # ─── abbreviation handling ───

    def test_abbreviations_dr(self):
        # D.C. followed by uppercase is genuinely ambiguous; 1 or 2 are both acceptable
        text = "Dr. Smith went to Washington D.C. He met Mr. Jones."
        sents = split_sentences(text)
        assert len(sents) in (1, 2), f"Expected 1-2 sentences, got {len(sents)}: {sents}"
        # Key check: "Dr." and "Mr." must NOT cause a split
        assert not any(s.strip() == "Dr." for s in sents), "Dr. was incorrectly split"
        assert not any(s.strip() == "Mr." for s in sents), "Mr. was incorrectly split"

    def test_abbreviations_eg(self):
        text = "Some items, e.g. apples and oranges, are fruits. They are healthy."
        sents = split_sentences(text)
        assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"

    def test_abbreviations_us(self):
        text = "The U.S. economy grew in 2024. This was unexpected."
        sents = split_sentences(text)
        assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"

    # ─── decimal / number handling ───

    def test_decimals(self):
        text = "The value is 3.14. It increased by $2.5M."
        sents = split_sentences(text)
        assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"

    def test_money(self):
        text = "Revenue reached $1.2B in Q3. Costs were $800M."
        sents = split_sentences(text)
        assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"

    # ─── bullet lists ───

    def test_bullet_dash(self):
        text = "- Item one\n- Item two\n- Item three"
        sents = split_sentences(text)
        assert len(sents) == 3, f"Expected 3 sentences, got {len(sents)}: {sents}"

    def test_bullet_numbered(self):
        text = "1. First point\n2. Second point\n3. Third point"
        sents = split_sentences(text)
        assert len(sents) == 3, f"Expected 3 sentences, got {len(sents)}: {sents}"

    def test_bullet_star(self):
        text = "* Alpha\n* Beta\n* Gamma"
        sents = split_sentences(text)
        assert len(sents) == 3, f"Expected 3 sentences, got {len(sents)}: {sents}"

    # ─── URLs and emails ───

    def test_url(self):
        text = "Visit https://example.com for details. It works well."
        sents = split_sentences(text)
        assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"

    def test_email(self):
        text = "Contact user@example.com for help. We respond quickly."
        sents = split_sentences(text)
        assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"

    # ─── mixed / complex ───

    def test_mixed_complex(self):
        text = "Dr. Smith earned $3.14M from the U.S. government. He donated it."
        sents = split_sentences(text)
        assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"

    def test_multiple_sentences(self):
        text = "Paris is the capital of France. Berlin is the capital of Germany. Tokyo is the capital of Japan."
        sents = split_sentences(text)
        assert len(sents) == 3, f"Expected 3 sentences, got {len(sents)}: {sents}"

    def test_exclamation_and_question(self):
        text = "What a day! Can you believe it? I certainly can."
        sents = split_sentences(text)
        assert len(sents) == 3, f"Expected 3 sentences, got {len(sents)}: {sents}"

    # ─── span correctness ───

    def test_spans_match_original_text(self):
        text = "First sentence. Second sentence. Third sentence."
        spans = split_sentences_with_spans(text)
        for sp in spans:
            extracted = text[sp["start"]:sp["end"]]
            assert extracted.strip() == sp["sentence"], (
                f"Span mismatch: got '{extracted}' vs '{sp['sentence']}'"
            )

    def test_spans_source_field(self):
        text = "- Bullet item\nRegular sentence here."
        spans = split_sentences_with_spans(text)
        assert any(s["source"] == "bullet" for s in spans), "Expected a bullet source"

    # ─── edge cases ───

    def test_empty_input(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []
        assert split_sentences_with_spans("") == []

    def test_single_sentence(self):
        text = "Just one sentence here."
        sents = split_sentences(text)
        assert len(sents) == 1

    def test_newline_separated_paragraphs(self):
        text = "First paragraph about topic A.\n\nSecond paragraph about topic B."
        sents = split_sentences(text)
        assert len(sents) >= 2, f"Expected >=2 sentences, got {len(sents)}: {sents}"

    def test_quotes(self):
        text = 'He said "Hello World." Then he left.'
        sents = split_sentences(text)
        # Should be 1 or 2 sentences, but NOT split inside the quotes incorrectly
        assert len(sents) <= 2, f"Too many splits: {sents}"


# ═══════════════════════════════════════════════════════════════
#  CLAIM FILTER TESTS
# ═══════════════════════════════════════════════════════════════

class TestClaimFilter:
    """Tests for is_claim heuristic."""

    # ─── obvious non-claims ───

    def test_filler_sure(self):
        assert is_claim("Sure!") is False

    def test_filler_of_course(self):
        assert is_claim("Of course.") is False

    def test_filler_great_question(self):
        assert is_claim("Great question!") is False

    def test_meta_let_me_explain(self):
        assert is_claim("Let me explain this in detail.") is False

    def test_meta_heres_what(self):
        assert is_claim("Here's what I found:") is False

    def test_meta_hope_this_helps(self):
        assert is_claim("Hope this helps!") is False

    def test_pure_question(self):
        assert is_claim("What do you think?") is False

    def test_short_fragment(self):
        assert is_claim("Yes.") is False

    def test_greeting(self):
        assert is_claim("Hello there!") is False

    # ─── obvious claims ───

    def test_factual_claim(self):
        assert is_claim("Paris is the capital of France.") is True

    def test_claim_with_number(self):
        assert is_claim("The population of Tokyo is 13.96 million.") is True

    def test_claim_with_date(self):
        assert is_claim("World War II ended in 1945.") is True

    def test_claim_causal(self):
        assert is_claim("Climate change has caused rising sea levels.") is True

    def test_claim_comparison(self):
        assert is_claim("Python is more popular than Ruby according to surveys.") is True

    def test_claim_financial(self):
        assert is_claim("Revenue increased by 25% in Q3 2024.") is True

    # ─── borderline cases (should be conservative → True) ───

    def test_borderline_kept(self):
        """Borderline sentences should be classified as claims (conservative)."""
        borderline = [
            "This is an interesting approach to the problem.",
            "The results suggest a strong correlation.",
            "It should work in most cases.",
        ]
        for sent in borderline:
            assert is_claim(sent) is True, f"Should be claim: '{sent}'"

    # ─── new patterns: meta/intro sentences ───

    def test_meta_ill_provide(self):
        assert is_claim("I'll provide an overview of various world religions.") is False

    def test_meta_ill_give(self):
        assert is_claim("I'll give you a brief summary.") is False

    def test_meta_ill_discuss(self):
        assert is_claim("I'll discuss the main points below.") is False

    def test_meta_let_me_provide(self):
        assert is_claim("Let me provide some context.") is False

    def test_meta_lets_explore(self):
        assert is_claim("Let's explore this topic in detail.") is False

    # ─── new patterns: advisory / caveat ───

    def test_advisory_keep_in_mind(self):
        assert is_claim("Keep in mind that every individual's perspective may vary within a faith tradition.") is False

    def test_advisory_may_vary(self):
        assert is_claim("Results may vary depending on your configuration.") is False

    def test_advisory_not_necessarily(self):
        assert is_claim("This is not necessarily true in all cases.") is False

    def test_advisory_it_depends(self):
        assert is_claim("It depends on the specific situation.") is False

    # ─── new patterns: label / heading with colon ───

    def test_label_colon(self):
        assert is_claim('Top contenders for "reasonable" religions:') is False

    def test_label_colon_short(self):
        assert is_claim("Key findings:") is False

    def test_label_colon_example_uses(self):
        assert is_claim("Example use cases:") is False

    def test_claim_with_colon_is_kept(self):
        """A sentence with a colon that's clearly a claim should stay."""
        assert is_claim("The ratio is 3:1 for every test case evaluated.") is True

    # ─── new patterns: hedging / opinion ───

    def test_hedging_i_think(self):
        assert is_claim("I think this approach is better.") is False

    def test_hedging_personally(self):
        assert is_claim("Personally, I prefer the first option.") is False

    def test_hedging_in_my_opinion(self):
        assert is_claim("In my opinion, React is easier to learn.") is False

    # ─── filter_claims batch function ───

    def test_filter_claims_batch(self):
        sentences = [
            "Sure!",
            "Paris is the capital of France.",
            "Let me explain.",
            "Revenue grew by 20%.",
        ]
        results = filter_claims(sentences)
        assert len(results) == 4
        assert results[0]["is_claim"] is False  # Sure!
        assert results[1]["is_claim"] is True   # Paris
        assert results[2]["is_claim"] is False  # Let me explain
        assert results[3]["is_claim"] is True   # Revenue


# ═══════════════════════════════════════════════════════════════
#  Run directly
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
