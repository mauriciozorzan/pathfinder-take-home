from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exact_item_ai.adjudicate import AdjudicationResult, NoopEvidenceAdjudicator
from exact_item_ai.main import run_pipeline
from exact_item_ai.models import ReceiptItem
from exact_item_ai.normalize import normalize_receipt_item, parse_price
from exact_item_ai.photo_assist import (
    AIBasedPhotoAnalyzer,
    BasicPhotoAnalyzer,
    NoopPhotoAnalyzer,
    PhotoAIConfig,
    PhotoEvidence,
    fetch_image_bytes,
)
from exact_item_ai.receipt_context import (
    NoopSiblingContextAdjudicator,
    NoopSharedPhotoAnalyzer,
    SiblingAdjudicationResult,
    SharedPhotoEvidence,
    VisiblePhotoCandidate,
    apply_receipt_level_context,
    compute_merchant_catalog_confidence,
    compute_receipt_coherence,
    compute_item_id_family_signal,
)
from exact_item_ai.resolve import ExactItemResolver
from exact_item_ai.route import route_item


class FakePhotoAnalyzer:
    def __init__(self, evidence: PhotoEvidence) -> None:
        self.evidence = evidence
        self.calls = 0

    def analyze(self, image_source: str, item_context: ReceiptItem) -> PhotoEvidence:
        self.calls += 1
        return self.evidence


class FakeAdjudicator:
    def __init__(self, result: AdjudicationResult) -> None:
        self.result = result
        self.calls = 0

    def adjudicate(
        self,
        item_context: ReceiptItem,
        current_result: object,
        photo_evidence: PhotoEvidence,
    ) -> AdjudicationResult:
        self.calls += 1
        return self.result


class FakeSharedPhotoAnalyzer:
    def __init__(self, evidence: SharedPhotoEvidence) -> None:
        self.evidence = evidence
        self.calls = 0

    def analyze_shared_photo(self, image_source: str, receipt_context: object) -> SharedPhotoEvidence:
        self.calls += 1
        return self.evidence


class FakeSiblingContextAdjudicator:
    def __init__(self, result: SiblingAdjudicationResult) -> None:
        self.result = result
        self.calls = 0

    def adjudicate(
        self,
        item_context: ReceiptItem,
        current_result: object,
        sibling_context: object,
    ) -> SiblingAdjudicationResult:
        self.calls += 1
        return self.result


class FakeHTTPResponse:
    def __init__(self, payload: bytes, *, content_type: str = "application/json", status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size is not None and size >= 0:
            return self.payload[:size]
        return self.payload


class NormalizeTests(unittest.TestCase):
    def test_parse_price(self) -> None:
        self.assertEqual(parse_price("$35.99"), 35.99)
        self.assertEqual(parse_price("799.00"), 799.0)
        self.assertIsNone(parse_price(None))

    def test_generic_titles_are_detected(self) -> None:
        item = ReceiptItem(
            dataset_name="test",
            receipt_index=0,
            item_index=0,
            merchant="Goodwill of Central & Northern Arizona",
            item_name="Books",
            item_id="2011857002992",
            item_price=2.99,
        )
        normalized = normalize_receipt_item(item)
        self.assertTrue(normalized.is_generic_title)
        self.assertEqual(normalized.normalized_merchant, "goodwill")

    def test_extracts_very_good_condition_from_resale_title(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="ThriftBooks",
                item_name="Salamandastron - Very Good condition",
                item_id=None,
                item_price=6.19,
            )
        )
        self.assertEqual(item.cleaned_item_name, "Salamandastron")
        self.assertEqual(item.normalized_item_name, "salamandastron")
        self.assertEqual(item.item_condition, "Very Good")
        self.assertIn("extracted condition: Very Good", item.normalization_notes)

    def test_extracts_good_condition_from_resale_title(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="ThriftBooks",
                item_name="Mossflower: A Tale of Redwall - Good condition",
                item_id=None,
                item_price=5.39,
            )
        )
        self.assertEqual(item.cleaned_item_name, "Mossflower: A Tale of Redwall")
        self.assertEqual(item.item_condition, "Good")

    def test_non_condition_titles_are_unchanged(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Bookshop",
                item_name="A Good Day for Climbing Trees",
                item_id=None,
                item_price=9.99,
            )
        )
        self.assertEqual(item.cleaned_item_name, "A Good Day for Climbing Trees")
        self.assertIsNone(item.item_condition)

    def test_strips_metadata_parenthetical_but_preserves_title(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Bambu Lab",
                item_name="Bambu Lab P2S - P2S Combo (Pre-sale, Ship around Apr.10, 2026)",
                item_id="1234",
                item_price=799.0,
            )
        )
        self.assertEqual(item.cleaned_item_name, "Bambu Lab P2S - P2S Combo")
        self.assertIn(
            "stripped metadata parenthetical: Pre-sale, Ship around Apr.10, 2026",
            item.normalization_notes,
        )

    def test_preserves_identity_parentheticals(self) -> None:
        size_item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Home Science Tools",
                item_name='Owl Pellets for Dissection, Large (1.5"+)',
                item_id=None,
                item_price=5.95,
            )
        )
        game_item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=1,
                merchant="Game Shop",
                item_name="Wavelength (board game)",
                item_id=None,
                item_price=35.99,
            )
        )
        self.assertIn('(1.5"+)', size_item.cleaned_item_name)
        self.assertIn("(board game)", game_item.cleaned_item_name)
        self.assertIn('preserved identity parenthetical: 1.5"+', size_item.normalization_notes)
        self.assertIn("preserved identity parenthetical: board game", game_item.normalization_notes)

    def test_weak_prefix_words_do_not_make_title_generic(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="Easter Basket Craft Kit",
                item_id=None,
                item_price=10.99,
            )
        )
        self.assertFalse(item.is_generic_title)
        self.assertFalse(item.looks_like_discount_line)

    def test_uncertain_merchant_alias_is_not_forced(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="MJ Designs USA",
                item_name="Craft Kit",
                item_id=None,
                item_price=10.99,
            )
        )
        self.assertEqual(item.normalized_merchant, "mj designs usa")

    def test_site_merch_is_weak_text_but_not_permanently_blocked_with_photo(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="SITE MERCH",
                item_id=None,
                item_price=10.99,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        self.assertTrue(item.is_generic_title)
        self.assertTrue(item.has_reference_photo)

    def test_normalization_does_not_expand_abbreviations_to_products(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="KA10 TT PEN",
                item_id=None,
                item_price=5.99,
            )
        )
        self.assertEqual(item.cleaned_item_name, "KA10 TT PEN")
        self.assertEqual(item.normalized_item_name, "ka10 tt pen")
        self.assertNotIn("twin", item.normalized_item_name)


class RoutingTests(unittest.TestCase):
    def test_specific_title_routes_to_deterministic(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Barnes & Noble Booksellers #2211",
                item_name="Wavelength by Alex Hague, Justin Vickers",
                item_id="0860001981704",
                item_price=35.99,
            )
        )
        decision = route_item(item)
        self.assertEqual(decision.bucket, "deterministic")

    def test_generic_title_routes_to_insufficient(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Goodwill of Central & Northern Arizona",
                item_name="Misc",
                item_id="2011846001999",
                item_price=1.99,
            )
        )
        decision = route_item(item)
        self.assertEqual(decision.bucket, "insufficient_evidence")


class ResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = ExactItemResolver(
            photo_analyzer=NoopPhotoAnalyzer(),
            adjudicator=NoopEvidenceAdjudicator(),
        )

    def test_resolves_book_like_items(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="ThriftBooks",
                item_name="Mattimeo: A Tale From Redwall - Very Good condition",
                item_id=None,
                item_price=6.69,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_entity_type, "book")
        self.assertEqual(result.condition, "Very Good")

    def test_resolves_short_condition_stripped_resale_book_title(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="ThriftBooks",
                item_name="Salamandastron - Very Good condition",
                item_id=None,
                item_price=6.19,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_title, "Salamandastron")
        self.assertEqual(result.route, "deterministic")
        self.assertEqual(result.evidence["initial_route"], "retrieval_needed")
        self.assertEqual(result.condition, "Very Good")

    def test_abstains_on_generic_thrift_line(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Goodwill of Central & Northern Arizona",
                item_name="Books",
                item_id="2011857002992",
                item_price=2.99,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.resolved_title)

    def test_retrieval_needed_item_stays_ambiguous_when_local_signal_is_weak(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Amazon",
                item_name="Rabbit",
                item_id=None,
                item_price=10.99,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.route, "retrieval_needed")
        self.assertNotIn("MVP", result.notes or "")
        self.assertNotIn("retrieval-backed resolver", result.notes or "")
        self.assertIn("plausible item family", result.notes or "")

    def test_retrieval_needed_note_explains_abbreviated_text(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="AB X",
                item_id=None,
                item_price=10.99,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertEqual(result.route, "retrieval_needed")
        self.assertEqual(result.status, "ambiguous")
        self.assertIn("abbreviated or truncated", result.notes or "")

    def test_retrieval_needed_note_mentions_catalog_lookup_for_weak_item_id(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="AB X",
                item_id="123456789012",
                item_price=10.99,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertEqual(result.route, "retrieval_needed")
        self.assertEqual(result.status, "ambiguous")
        self.assertIn("catalog or lookup evidence", result.notes or "")

    def test_high_specificity_named_work_resolves_without_retrieval_hit(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Online Retailer",
                item_name="The Gentle Fox Listened",
                item_id=None,
                item_price=10.99,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_title, "The Gentle Fox Listened")
        self.assertEqual(result.route, "deterministic")
        self.assertEqual(result.evidence["initial_route"], "retrieval_needed")
        self.assertEqual(result.evidence["candidate_basis"], "high_specificity_title")

    def test_high_specificity_branded_product_title_resolves(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="Little Maker Adventure Rocket",
                item_id=None,
                item_price=109.99,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.route, "deterministic")
        self.assertEqual(result.evidence["candidate_basis"], "high_specificity_title")

    def test_specific_but_common_category_title_can_stay_ambiguous(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Target",
                item_name="Eight Ball Helmet",
                item_id=None,
                item_price=27.99,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.route, "retrieval_needed")

    def test_created_by_phrase_is_not_misread_as_book_authorship(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Amazon",
                item_name="Melissa & Doug Created by Me! Monster Truck Wooden Craft Kit",
                item_id=None,
                item_price=10.99,
            )
        )
        result = self.resolver.resolve_item(item)
        self.assertNotEqual(result.resolved_entity_type, "book")

    def test_photo_path_not_triggered_for_deterministic_items(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=True,
                success=True,
                summary="Should not be used.",
                confidence_delta=0.1,
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=NoopEvidenceAdjudicator())
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Barnes & Noble Booksellers #2211",
                item_name="Wavelength by Alex Hague, Justin Vickers",
                item_id="0860001981704",
                item_price=35.99,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertEqual(result.status, "resolved")
        self.assertFalse(result.photo_analysis_attempted)
        self.assertEqual(analyzer.calls, 0)

    def test_photo_path_triggered_for_eligible_anchored_item(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=True,
                success=True,
                summary="Packaging suggests a Nintendo Switch Minecraft product.",
                extracted_signals=["minecraft branding", "switch case"],
                suggested_title="Minecraft - Nintendo Switch",
                suggested_description="Minecraft video game for Nintendo Switch.",
                confidence_delta=0.18,
                notes="Test analyzer matched image evidence.",
                analyzer_name="fake",
                model_confidence=0.9,
                is_sufficient_for_exact_identification=True,
            )
        )
        adjudicator = FakeAdjudicator(
            AdjudicationResult(
                should_use_photo_result=True,
                should_refine_existing_result=False,
                final_status="resolved",
                final_confidence=0.82,
                adjudicated_title="Minecraft Touch Screen Interactive Smartwatch",
                adjudicated_description="Minecraft-branded interactive smart watch.",
                contradiction_detected=False,
                rationale="Model judged the image-derived identity as a plausible expansion of the receipt shorthand.",
                evidence_summary=["photo title visible", "receipt shorthand plausible"],
                decision="use_photo_result",
                plausible_refinement=True,
                contradiction_strength="none",
                image_as_high_weight_evidence=True,
                refinement_rationale="Receipt text is a shorthand label and image title is more specific.",
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=adjudicator)
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Walmart",
                item_name="MINECRAFT SW",
                item_id="030506649600",
                item_price=29.98,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertTrue(result.photo_analysis_attempted)
        self.assertTrue(result.photo_analysis_success)
        self.assertEqual(result.resolved_title, "Minecraft Touch Screen Interactive Smartwatch")
        self.assertEqual(result.status, "resolved")
        self.assertGreater(result.photo_confidence_delta, 0.0)
        self.assertEqual(analyzer.calls, 1)
        self.assertEqual(adjudicator.calls, 1)
        self.assertEqual(result.adjudication_decision, "use_photo_result")

    def test_basic_analyzer_attempts_photo_path_without_forcing_result(self) -> None:
        resolver = ExactItemResolver(
            photo_analyzer=BasicPhotoAnalyzer(probe_sources=False),
            adjudicator=NoopEvidenceAdjudicator(),
        )
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Walmart",
                item_name="MINECRAFT SW",
                item_id="030506649600",
                item_price=29.98,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertTrue(result.photo_analysis_attempted)
        self.assertFalse(result.photo_analysis_success)
        self.assertFalse(result.photo_evidence_used)
        self.assertEqual(result.status, "ambiguous")

    def test_noop_analyzer_does_not_change_results(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Walmart",
                item_name="MINECRAFT SW",
                item_id="030506649600",
                item_price=29.98,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        result = resolver.resolve_item(item)
        self.assertTrue(result.photo_analysis_attempted)
        self.assertFalse(result.photo_analysis_success)
        self.assertFalse(result.photo_result_changed)
        self.assertEqual(result.status, "ambiguous")

    def test_photo_can_increase_confidence_without_forcing_resolution(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=True,
                success=True,
                summary="Photo hints at a book cover but title remains partially obscured.",
                extracted_signals=["book cover"],
                confidence_delta=0.07,
                notes="Inconclusive but somewhat supportive.",
                analyzer_name="fake",
                model_confidence=0.62,
                is_sufficient_for_exact_identification=False,
            )
        )
        adjudicator = FakeAdjudicator(
            AdjudicationResult(
                should_use_photo_result=False,
                should_refine_existing_result=True,
                final_status="ambiguous",
                final_confidence=0.56,
                adjudicated_title="Partially visible book title",
                adjudicated_description=None,
                contradiction_detected=False,
                rationale="Photo evidence is useful but not enough to resolve.",
                evidence_summary=["partial cover evidence"],
                decision="refine_existing_result",
                plausible_refinement=True,
                contradiction_strength="none",
                image_as_high_weight_evidence=True,
                refinement_rationale="Photo partially clarifies the item.",
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=adjudicator)
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Walmart",
                item_name="SITE MERCH",
                item_id="978006348938",
                item_price=23.78,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertTrue(result.photo_analysis_attempted)
        self.assertEqual(result.status, "ambiguous")
        self.assertGreater(result.confidence, 0.49)
        self.assertTrue(result.photo_result_changed)
        self.assertEqual(result.adjudication_decision, "refine_existing_result")

    def test_abstention_still_works_when_photo_is_inconclusive(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=False,
                success=False,
                summary=None,
                confidence_delta=0.0,
                notes="Photo too blurry to use.",
                analyzer_name="fake",
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=NoopEvidenceAdjudicator())
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Walmart",
                item_name="KA10 TT PEN",
                item_id="081968202851",
                item_price=5.97,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertEqual(result.status, "ambiguous")
        self.assertTrue(result.photo_analysis_attempted)
        self.assertFalse(result.photo_analysis_success)
        self.assertFalse(result.photo_result_changed)

    def test_contradiction_from_adjudicator_fails_closed(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=True,
                success=True,
                summary="Image appears unrelated to receipt text.",
                extracted_signals=["different product"],
                suggested_title="Unrelated Product",
                confidence_delta=0.1,
                analyzer_name="fake",
                model_confidence=0.9,
                is_sufficient_for_exact_identification=True,
            )
        )
        adjudicator = FakeAdjudicator(
            AdjudicationResult(
                should_use_photo_result=False,
                should_refine_existing_result=False,
                final_status="ambiguous",
                final_confidence=0.3,
                adjudicated_title=None,
                adjudicated_description=None,
                contradiction_detected=True,
                rationale="Receipt context and photo evidence appear contradictory.",
                evidence_summary=["contradiction"],
                decision="contradiction",
                plausible_refinement=False,
                contradiction_strength="strong",
                image_as_high_weight_evidence=True,
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=adjudicator)
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Walmart",
                item_name="MINECRAFT SW",
                item_id="030506649600",
                item_price=29.98,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertEqual(result.status, "ambiguous")
        self.assertFalse(result.photo_evidence_used)
        self.assertTrue(result.adjudication_contradiction_detected)
        self.assertEqual(result.adjudication_contradiction_strength, "strong")

    def test_plausible_image_refinement_resolves_without_item_specific_rules(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=True,
                success=True,
                summary="Image shows visible packaging and a more specific retail product title.",
                extracted_signals=["visible title", "retail packaging", "brand visible"],
                suggested_title="Specific Product Title With Receipt Token",
                suggested_description="Specific packaged retail product identified from image.",
                confidence_delta=0.12,
                analyzer_name="fake",
                model_confidence=0.91,
                is_sufficient_for_exact_identification=True,
            )
        )
        adjudicator = FakeAdjudicator(
            AdjudicationResult(
                should_use_photo_result=True,
                should_refine_existing_result=False,
                final_status="resolved",
                final_confidence=0.84,
                adjudicated_title="Specific Product Title With Receipt Token",
                adjudicated_description="Specific packaged retail product identified from image.",
                contradiction_detected=False,
                rationale="Receipt text is shorthand; the image-derived title plausibly explains it.",
                evidence_summary=["image has title", "receipt token is included"],
                decision="use_photo_result",
                plausible_refinement=True,
                contradiction_strength="none",
                image_as_high_weight_evidence=True,
                refinement_rationale="Anchored image gives a more specific exact title.",
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=adjudicator)
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="TOKEN X",
                item_id="123456789012",
                item_price=12.99,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_title, "Specific Product Title With Receipt Token")
        self.assertTrue(result.adjudication_plausible_refinement)

    def test_weak_contradiction_does_not_block_plausible_refinement(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=True,
                success=True,
                summary="Image shows a packaged item with a more specific product title.",
                extracted_signals=["packaging", "visible title"],
                suggested_title="Specific Packaged Product",
                suggested_description="Specific product from anchored packaging.",
                confidence_delta=0.12,
                analyzer_name="fake",
                model_confidence=0.88,
                is_sufficient_for_exact_identification=True,
            )
        )
        adjudicator = FakeAdjudicator(
            AdjudicationResult(
                should_use_photo_result=True,
                should_refine_existing_result=False,
                final_status="resolved",
                final_confidence=0.8,
                adjudicated_title="Specific Packaged Product",
                adjudicated_description="Specific product from anchored packaging.",
                contradiction_detected=True,
                rationale="There is a weak category wording mismatch, but the image plausibly explains shorthand.",
                evidence_summary=["weak wording mismatch", "photo is high-weight evidence"],
                decision="use_photo_result",
                plausible_refinement=True,
                contradiction_strength="weak",
                image_as_high_weight_evidence=True,
                refinement_rationale="Receipt wording is shorthand, not strict category truth.",
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=adjudicator)
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="GENERIC X",
                item_id="123456789012",
                item_price=12.99,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.adjudication_contradiction_strength, "weak")

    def test_strong_photo_candidate_reaches_adjudication_even_if_analyzer_does_not_use_it(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=False,
                success=True,
                summary="Image shows a specific packaged product with visible title and brand.",
                extracted_signals=["visible title", "brand visible", "10 pack"],
                suggested_title="Brand Twin Tip Metallic Pens",
                suggested_description="Specific packaged art supply product.",
                confidence_delta=0.06,
                analyzer_name="fake",
                model_confidence=0.9,
                model_suggested_title="Brand Twin Tip Metallic Pens",
                model_suggested_description="Specific packaged art supply product.",
                model_notes="Receipt shorthand is not an exact literal match.",
                is_sufficient_for_exact_identification=False,
            )
        )
        adjudicator = FakeAdjudicator(
            AdjudicationResult(
                should_use_photo_result=True,
                should_refine_existing_result=False,
                final_status="resolved",
                final_confidence=0.86,
                adjudicated_title="Brand Twin Tip Metallic Pens",
                adjudicated_description="Specific packaged art supply product.",
                contradiction_detected=False,
                rationale="The photo-derived title plausibly explains the abbreviated receipt line.",
                evidence_summary=["strong photo title", "receipt shorthand is compatible"],
                decision="use_photo_result",
                plausible_refinement=True,
                contradiction_strength="none",
                image_as_high_weight_evidence=True,
                refinement_rationale="Adjudication, not pre-filtering, decides the semantic match.",
                photo_refinement_strength="strong",
                supports_exact_resolution=True,
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=adjudicator)
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="ABBR X",
                item_id="123456789012",
                item_price=5.99,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertEqual(adjudicator.calls, 1)
        self.assertTrue(result.photo_candidate_specific)
        self.assertTrue(result.photo_candidate_sent_to_adjudication)
        self.assertIsNone(result.photo_adjudication_block_reason)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_title, "Brand Twin Tip Metallic Pens")

    def test_weak_unused_photo_candidate_can_be_blocked_before_adjudication(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=False,
                success=True,
                summary="Image gives only a partial clue.",
                extracted_signals=["partial label"],
                suggested_title="Possible Item",
                confidence_delta=0.02,
                analyzer_name="fake",
                model_confidence=0.62,
                model_suggested_title="Possible Item",
                is_sufficient_for_exact_identification=False,
            )
        )
        adjudicator = FakeAdjudicator(
            AdjudicationResult(
                should_use_photo_result=True,
                should_refine_existing_result=False,
                final_status="resolved",
                final_confidence=0.9,
                adjudicated_title="Should Not Be Used",
                adjudicated_description=None,
                contradiction_detected=False,
                rationale="Should not be called.",
                evidence_summary=[],
                decision="use_photo_result",
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=adjudicator)
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="ABBR X",
                item_id="123456789012",
                item_price=5.99,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertEqual(adjudicator.calls, 0)
        self.assertFalse(result.photo_candidate_specific)
        self.assertFalse(result.photo_candidate_sent_to_adjudication)
        self.assertIsNotNone(result.photo_adjudication_block_reason)

    def test_strong_photo_refinement_can_resolve_even_when_adjudicator_is_cautious(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=True,
                success=True,
                summary="Image shows visible branded packaging and a full product title.",
                extracted_signals=["visible title", "brand visible", "packaging"],
                suggested_title="Brand Unicorn Delivery Set",
                suggested_description="Branded packaged retail product.",
                confidence_delta=0.12,
                analyzer_name="fake",
                model_confidence=0.9,
                model_suggested_title="Brand Unicorn Delivery Set",
                model_suggested_description="Branded packaged retail product.",
                is_sufficient_for_exact_identification=True,
            )
        )
        adjudicator = FakeAdjudicator(
            AdjudicationResult(
                should_use_photo_result=False,
                should_refine_existing_result=True,
                final_status="ambiguous",
                final_confidence=0.6,
                adjudicated_title="Brand Unicorn Delivery Set",
                adjudicated_description="Branded packaged retail product.",
                contradiction_detected=False,
                rationale="Photo title plausibly explains the receipt shorthand.",
                evidence_summary=["visible packaging"],
                decision="refine_existing_result",
                plausible_refinement=True,
                contradiction_strength="none",
                image_as_high_weight_evidence=True,
                refinement_rationale="The image provides the full specific title.",
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=adjudicator)
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="UNICORN SHORT",
                item_id=None,
                item_price=9.99,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_title, "Brand Unicorn Delivery Set")
        self.assertTrue(result.photo_evidence_used)

    def test_weak_photo_clue_does_not_force_resolution(self) -> None:
        analyzer = FakePhotoAnalyzer(
            PhotoEvidence(
                used=True,
                success=True,
                summary="Image gives a possible clue but not a grounded exact item.",
                extracted_signals=["partial packaging"],
                suggested_title="Possible Product",
                confidence_delta=0.05,
                analyzer_name="fake",
                model_confidence=0.62,
                model_suggested_title="Possible Product",
                is_sufficient_for_exact_identification=False,
            )
        )
        adjudicator = FakeAdjudicator(
            AdjudicationResult(
                should_use_photo_result=False,
                should_refine_existing_result=True,
                final_status="ambiguous",
                final_confidence=0.55,
                adjudicated_title="Possible Product",
                adjudicated_description=None,
                contradiction_detected=False,
                rationale="Photo is helpful but weak.",
                evidence_summary=["partial packaging"],
                decision="refine_existing_result",
                plausible_refinement=True,
                contradiction_strength="none",
                image_as_high_weight_evidence=False,
                photo_refinement_strength="weak",
                supports_exact_resolution=False,
            )
        )
        resolver = ExactItemResolver(photo_analyzer=analyzer, adjudicator=adjudicator)
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Retailer",
                item_name="WEAK SHORT",
                item_id=None,
                item_price=9.99,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        result = resolver.resolve_item(item)
        self.assertEqual(result.status, "ambiguous")

    def test_no_product_specific_mappings_exist_in_photo_module(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "exact_item_ai" / "photo_assist.py").read_text()
        self.assertNotIn("minecraft sw", source.lower())
        self.assertNotIn("ward d pape", source.lower())
        self.assertNotIn("nintendo switch", source.lower())


class ReceiptContextTests(unittest.TestCase):
    def test_catalog_context_promotes_named_item_with_id_in_coherent_receipt(self) -> None:
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        items = [
            normalize_receipt_item(ReceiptItem("test", 0, 0, "Catalog Merchant", "Course Book: Level 5", "1001", 20.0)),
            normalize_receipt_item(ReceiptItem("test", 0, 1, "Catalog Merchant", "Answer Key: Level 5", "1002", 10.0)),
            normalize_receipt_item(ReceiptItem("test", 0, 2, "Catalog Merchant", "Handwriting Workbook: Level 5", "1003", 12.0)),
            normalize_receipt_item(ReceiptItem("test", 0, 3, "Catalog Merchant", "Ice Storm", "1004", 7.0)),
        ]
        base_results = [resolver.resolve_item(item) for item in items]
        self.assertEqual(base_results[3].route, "retrieval_needed")
        self.assertEqual(base_results[3].status, "ambiguous")

        updated = apply_receipt_level_context(
            items,
            base_results,
            shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
            sibling_context_adjudicator=NoopSiblingContextAdjudicator(),
        )
        self.assertEqual(updated[3].route, "deterministic")
        self.assertEqual(updated[3].status, "resolved")
        self.assertEqual(updated[3].resolved_title, "Ice Storm")
        self.assertEqual(updated[3].receipt_level_assignment_basis, "catalog_context_promotion")

    def test_catalog_context_does_not_promote_generic_labels(self) -> None:
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        items = [
            normalize_receipt_item(ReceiptItem("test", 0, 0, "Catalog Merchant", "Course Book: Level 5", "1001", 20.0)),
            normalize_receipt_item(ReceiptItem("test", 0, 1, "Catalog Merchant", "Answer Key: Level 5", "1002", 10.0)),
            normalize_receipt_item(ReceiptItem("test", 0, 2, "Catalog Merchant", "Handwriting Workbook: Level 5", "1003", 12.0)),
            normalize_receipt_item(ReceiptItem("test", 0, 3, "Catalog Merchant", "Books", "1004", 7.0)),
        ]
        updated = apply_receipt_level_context(
            items,
            [resolver.resolve_item(item) for item in items],
            shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
            sibling_context_adjudicator=NoopSiblingContextAdjudicator(),
        )
        self.assertNotEqual(updated[3].status, "resolved")
        self.assertFalse(updated[3].receipt_level_assignment_used)

    def test_catalog_confidence_and_coherence_are_general_receipt_features(self) -> None:
        items = [
            normalize_receipt_item(ReceiptItem("test", 0, 0, "Any Catalog", "Course Book: Level 5", "1001", 20.0)),
            normalize_receipt_item(ReceiptItem("test", 0, 1, "Any Catalog", "Answer Key: Level 5", "1002", 10.0)),
            normalize_receipt_item(ReceiptItem("test", 0, 2, "Any Catalog", "Handwriting Workbook: Level 5", "1003", 12.0)),
            normalize_receipt_item(ReceiptItem("test", 0, 3, "Any Catalog", "Ice Storm", "1004", 7.0)),
        ]
        self.assertGreaterEqual(compute_merchant_catalog_confidence(items, "Any Catalog"), 0.68)
        self.assertGreaterEqual(compute_receipt_coherence(items), 0.62)

    def test_sibling_context_can_raise_confidence_without_resolving(self) -> None:
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        sibling = normalize_receipt_item(
            ReceiptItem("test", 0, 0, "Retailer", "Red Gadget", None, 4.99)
        )
        weak_item = normalize_receipt_item(
            ReceiptItem("test", 0, 1, "Retailer", "Blue Widget", None, 4.99)
        )
        ambiguous_result = replace(
            resolver.resolve_item(weak_item),
            status="ambiguous",
            resolved_title="Blue Widget",
            resolved_description="Plausible but not fully verified product.",
            resolved_entity_type="product",
            confidence=0.54,
        )
        sibling_result = replace(
            resolver.resolve_item(sibling),
            status="resolved",
            resolved_title="Red Gadget",
            resolved_description="Resolved sibling product.",
            resolved_entity_type="product",
            confidence=0.8,
        )
        updated = apply_receipt_level_context(
            [sibling, weak_item],
            [sibling_result, ambiguous_result],
            shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
            sibling_context_adjudicator=FakeSiblingContextAdjudicator(
                SiblingAdjudicationResult(
                    family_consistent=True,
                    sibling_support_strength="weak",
                    should_increase_confidence=True,
                    confidence_delta=0.04,
                    can_promote_to_resolved=False,
                    rationale="Sibling context is helpful but not decisive.",
                    adjudicator_name="fake",
                )
            ),
        )
        self.assertEqual(updated[1].status, "ambiguous")
        self.assertGreater(updated[1].confidence, ambiguous_result.confidence)
        self.assertTrue(updated[1].sibling_context_used)

    def test_sibling_context_alone_does_not_force_resolution(self) -> None:
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        sibling = normalize_receipt_item(
            ReceiptItem("test", 0, 0, "Retailer", "Specific Product Name", "123456789010", 4.99)
        )
        generic = normalize_receipt_item(
            ReceiptItem("test", 0, 1, "Retailer", "Misc", "123456789011", 4.99)
        )
        updated = apply_receipt_level_context(
            [sibling, generic],
            [resolver.resolve_item(sibling), resolver.resolve_item(generic)],
            shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
            sibling_context_adjudicator=FakeSiblingContextAdjudicator(
                SiblingAdjudicationResult(
                    family_consistent=True,
                    sibling_support_strength="strong",
                    should_increase_confidence=True,
                    confidence_delta=0.2,
                    can_promote_to_resolved=True,
                    rationale="Sibling context is strong, but there is no current candidate.",
                    adjudicator_name="fake",
                )
            ),
        )
        self.assertNotEqual(updated[1].status, "resolved")
        self.assertFalse(updated[1].sibling_context_used)

    def test_item_id_similarity_is_only_weak_family_signal(self) -> None:
        item = normalize_receipt_item(
            ReceiptItem("test", 0, 0, "Retailer", "Item A", "123456789010", 1.0)
        )
        sibling = normalize_receipt_item(
            ReceiptItem("test", 0, 1, "Retailer", "Item B", "123456789012", 1.0)
        )
        signals = compute_item_id_family_signal(item, [sibling])
        self.assertIn("same_prefix", signals)
        self.assertIn("possibly_related_family", signals)

    def test_shared_photo_candidates_can_resolve_generic_leftover(self) -> None:
        shared_url = "https://example.com/shared.jpg"
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        named = normalize_receipt_item(
            ReceiptItem(
                "test",
                0,
                0,
                "Book Shop",
                "Math Workbook",
                None,
                8.0,
                reference_photo_urls=[shared_url],
            )
        )
        generic = normalize_receipt_item(
            ReceiptItem(
                "test",
                0,
                1,
                "Book Shop",
                "Used Curriculum and Books",
                None,
                4.0,
                reference_photo_urls=[shared_url],
            )
        )
        analyzer = FakeSharedPhotoAnalyzer(
            SharedPhotoEvidence(
                photo_url=shared_url,
                success=True,
                analyzer_name="fake_shared",
                candidates=[
                    VisiblePhotoCandidate("Math Workbook", "Visible math workbook.", "book", None, 0.9, ["title visible"]),
                    VisiblePhotoCandidate("Science Reader", "Visible science book.", "book", None, 0.86, ["book cover visible"]),
                ],
            )
        )
        updated = apply_receipt_level_context(
            [named, generic],
            [resolver.resolve_item(named), resolver.resolve_item(generic)],
            shared_photo_analyzer=analyzer,
            sibling_context_adjudicator=NoopSiblingContextAdjudicator(),
        )
        self.assertEqual(analyzer.calls, 1)
        self.assertEqual(updated[1].status, "resolved")
        self.assertEqual(updated[1].route, "photo_assisted")
        self.assertEqual(updated[1].resolved_title, "Science Reader")
        self.assertIn("initial_route", updated[1].evidence)
        self.assertTrue(updated[1].shared_photo_used)
        self.assertEqual(updated[1].receipt_level_assignment_basis, "shared_photo_leftover_generic_line")

    def test_shared_photo_incoherent_leftovers_remain_unresolved(self) -> None:
        shared_url = "https://example.com/shared.jpg"
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        generic = normalize_receipt_item(
            ReceiptItem("test", 0, 0, "Book Shop", "Used Curriculum and Books", None, 4.0, reference_photo_urls=[shared_url])
        )
        other_generic = normalize_receipt_item(
            ReceiptItem("test", 0, 1, "Book Shop", "Books", None, 3.0, reference_photo_urls=[shared_url])
        )
        analyzer = FakeSharedPhotoAnalyzer(
            SharedPhotoEvidence(
                photo_url=shared_url,
                success=True,
                analyzer_name="fake_shared",
                candidates=[
                    VisiblePhotoCandidate("Science Reader", None, "book", None, 0.86, []),
                    VisiblePhotoCandidate("History Reader", None, "book", None, 0.84, []),
                ],
            )
        )
        updated = apply_receipt_level_context(
            [generic, other_generic],
            [resolver.resolve_item(generic), resolver.resolve_item(other_generic)],
            shared_photo_analyzer=analyzer,
            sibling_context_adjudicator=NoopSiblingContextAdjudicator(),
        )
        self.assertNotEqual(updated[0].status, "resolved")
        self.assertNotEqual(updated[1].status, "resolved")

    def test_sibling_family_context_can_resolve_plausible_ambiguous_item(self) -> None:
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        ambiguous_item = normalize_receipt_item(
            ReceiptItem("test", 0, 0, "Retailer", "Eight Ball Helmet", None, 27.99)
        )
        sibling_item = normalize_receipt_item(
            ReceiptItem("test", 0, 1, "Retailer", "Bell Indy Adult Bike Helmet", None, 27.99)
        )
        updated = apply_receipt_level_context(
            [ambiguous_item, sibling_item],
            [resolver.resolve_item(ambiguous_item), resolver.resolve_item(sibling_item)],
            shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
            sibling_context_adjudicator=FakeSiblingContextAdjudicator(
                SiblingAdjudicationResult(
                    family_consistent=True,
                    sibling_support_strength="strong",
                    should_increase_confidence=True,
                    confidence_delta=0.2,
                    can_promote_to_resolved=True,
                    rationale="The model judges the sibling as same-family support.",
                    notes=["Resolved sibling strengthens the current candidate."],
                    adjudicator_name="fake",
                )
            ),
        )
        self.assertEqual(updated[0].status, "resolved")
        self.assertEqual(updated[0].resolved_title, "Eight Ball Helmet")
        self.assertTrue(updated[0].sibling_context_used)
        self.assertTrue(updated[0].family_consistent_with_siblings)
        self.assertTrue(updated[0].sibling_context_changed_status)
        self.assertEqual(updated[0].sibling_similarity_score, 0.9)

    def test_sibling_context_can_resolve_photo_candidate_with_family_support(self) -> None:
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        ambiguous_item = normalize_receipt_item(
            ReceiptItem("test", 0, 0, "Retailer", "KA10 TT PEN", "081968202851", 5.97)
        )
        sibling_item = normalize_receipt_item(
            ReceiptItem("test", 0, 1, "Retailer", "KA12GLPNS", "081968202871", 3.98)
        )
        ambiguous_result = replace(
            resolver.resolve_item(ambiguous_item),
            photo_model_suggested_title="Kingart Twin-Tip Metallic Pens - 10 Pack",
            photo_model_suggested_description="A set of 10 metallic pens.",
            photo_model_confidence=0.9,
        )
        sibling_result = replace(
            resolver.resolve_item(sibling_item),
            status="resolved",
            resolved_title="Kingart Glitter Gel Pens - 12 Pack",
            resolved_description="A set of glitter gel pens.",
            resolved_entity_type="product",
            confidence=0.9,
        )
        updated = apply_receipt_level_context(
            [ambiguous_item, sibling_item],
            [ambiguous_result, sibling_result],
            shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
            sibling_context_adjudicator=FakeSiblingContextAdjudicator(
                SiblingAdjudicationResult(
                    family_consistent=True,
                    sibling_support_strength="strong",
                    should_increase_confidence=True,
                    confidence_delta=0.2,
                    can_promote_to_resolved=True,
                    rationale="The model judges the photo candidate as consistent with the resolved sibling.",
                    adjudicator_name="fake",
                )
            ),
        )
        self.assertEqual(updated[0].status, "resolved")
        self.assertEqual(updated[0].resolved_title, "Kingart Twin-Tip Metallic Pens - 10 Pack")
        self.assertTrue(updated[0].sibling_context_used)
        self.assertTrue(updated[0].photo_evidence_used)

    def test_nearby_item_ids_alone_do_not_promote(self) -> None:
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        ambiguous_item = normalize_receipt_item(
            ReceiptItem("test", 0, 0, "Retailer", "Blue Thing", "123456789010", 5.0)
        )
        sibling_item = normalize_receipt_item(
            ReceiptItem("test", 0, 1, "Retailer", "Red Object", "123456789012", 5.0)
        )
        ambiguous_result = replace(
            resolver.resolve_item(ambiguous_item),
            status="ambiguous",
            resolved_title=None,
            resolved_description=None,
            resolved_entity_type=None,
            evidence={"used_title_match": False, "candidate_basis": "needs_retrieval"},
        )
        sibling_result = replace(
            resolver.resolve_item(sibling_item),
            status="resolved",
            resolved_title="Red Object",
            resolved_description="Resolved sibling item.",
            resolved_entity_type="product",
            confidence=0.9,
        )
        updated = apply_receipt_level_context(
            [ambiguous_item, sibling_item],
            [ambiguous_result, sibling_result],
            shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
            sibling_context_adjudicator=FakeSiblingContextAdjudicator(
                SiblingAdjudicationResult(
                    family_consistent=True,
                    sibling_support_strength="strong",
                    should_increase_confidence=True,
                    confidence_delta=0.2,
                    can_promote_to_resolved=True,
                    rationale="The model output is strong, but there is no current candidate.",
                    adjudicator_name="fake",
                )
            ),
        )
        self.assertNotEqual(updated[0].status, "resolved")

    def test_sibling_context_does_not_override_strong_contradiction(self) -> None:
        resolver = ExactItemResolver(photo_analyzer=NoopPhotoAnalyzer(), adjudicator=NoopEvidenceAdjudicator())
        ambiguous_item = normalize_receipt_item(
            ReceiptItem("test", 0, 0, "Retailer", "Short Pen", "123456789010", 5.0)
        )
        sibling_item = normalize_receipt_item(
            ReceiptItem("test", 0, 1, "Retailer", "Resolved Gel Pen Set", "123456789012", 5.0)
        )
        ambiguous_result = replace(
            resolver.resolve_item(ambiguous_item),
            status="ambiguous",
            photo_model_suggested_title="Gel Pen Set",
            photo_model_confidence=0.9,
            adjudication_contradiction_detected=True,
            adjudication_contradiction_strength="strong",
        )
        sibling_result = replace(
            resolver.resolve_item(sibling_item),
            status="resolved",
            resolved_title="Resolved Gel Pen Set",
            confidence=0.9,
        )
        updated = apply_receipt_level_context(
            [ambiguous_item, sibling_item],
            [ambiguous_result, sibling_result],
            shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
            sibling_context_adjudicator=FakeSiblingContextAdjudicator(
                SiblingAdjudicationResult(
                    family_consistent=True,
                    sibling_support_strength="strong",
                    should_increase_confidence=True,
                    confidence_delta=0.2,
                    can_promote_to_resolved=True,
                    rationale="The model output is strong, but direct evidence has a strong contradiction.",
                    adjudicator_name="fake",
                )
            ),
        )
        self.assertEqual(updated[0].status, "ambiguous")

    def test_sibling_context_uses_model_adjudication_not_hardcoded_family_maps(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "exact_item_ai" / "receipt_context.py").read_text()
        self.assertNotIn("_product_family_tokens", source)
        self.assertNotIn("_category_hint", source)
        self.assertNotIn("compute_sibling_similarity", source)
        self.assertNotIn("generic_descriptors", source)


class AIBasedPhotoAnalyzerTests(unittest.TestCase):
    def test_fetch_image_bytes_rejects_unsupported_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "not_image.txt"
            path.write_text("not an image")
            result = fetch_image_bytes(str(path), timeout_seconds=1.0, max_image_bytes=1000)
        self.assertFalse(result.success)
        self.assertIn("Unsupported image", result.error or "")

    def test_ai_analyzer_disabled_is_safe_noop(self) -> None:
        analyzer = AIBasedPhotoAnalyzer(
            PhotoAIConfig(enabled=False, api_key="test-key")
        )
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Walmart",
                item_name="MINECRAFT SW",
                item_id="030506649600",
                item_price=29.98,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        evidence = analyzer.analyze("https://example.com/photo.jpg", item)
        self.assertFalse(evidence.used)
        self.assertFalse(evidence.success)

    def test_ai_analyzer_fetch_failure_is_safe_noop(self) -> None:
        analyzer = AIBasedPhotoAnalyzer(
            PhotoAIConfig(enabled=True, api_key="test-key", timeout_seconds=1.0)
        )
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Walmart",
                item_name="MINECRAFT SW",
                item_id="030506649600",
                item_price=29.98,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        with patch("exact_item_ai.photo_assist.urlopen", side_effect=OSError("network down")):
            evidence = analyzer.analyze("https://example.com/photo.jpg", item)
        self.assertFalse(evidence.used)
        self.assertFalse(evidence.success)
        self.assertIn("Image fetch failed", evidence.notes or "")

    def test_ai_analyzer_parses_structured_model_response(self) -> None:
        analyzer = AIBasedPhotoAnalyzer(
            PhotoAIConfig(enabled=True, api_key="test-key", timeout_seconds=1.0)
        )
        item = normalize_receipt_item(
            ReceiptItem(
                dataset_name="test",
                receipt_index=0,
                item_index=0,
                merchant="Walmart",
                item_name="MINECRAFT SW",
                item_id="030506649600",
                item_price=29.98,
                reference_photo_urls=["https://example.com/photo.jpg"],
            )
        )
        model_json = {
            "visible_title": "Kids Smart Watch",
            "product_type": "smart watch",
            "brand_or_author": None,
            "manufacturer_or_publisher": None,
            "suggested_title": "Kids Smart Watch",
            "suggested_description": "Children's smart watch product visible in packaging.",
            "confidence": 0.86,
            "is_sufficient_for_exact_identification": True,
            "signals": ["smart watch packaging", "watch image visible"],
            "notes": "Image clearly shows smartwatch packaging.",
        }
        chat_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(model_json),
                    }
                }
            ]
        }
        with patch(
            "exact_item_ai.photo_assist.urlopen",
            side_effect=[
                FakeHTTPResponse(b"fake-image-bytes", content_type="image/jpeg"),
                FakeHTTPResponse(json.dumps(chat_payload).encode("utf-8")),
            ],
        ):
            evidence = analyzer.analyze("https://example.com/photo.jpg", item)
        self.assertTrue(evidence.used)
        self.assertTrue(evidence.success)
        self.assertEqual(evidence.suggested_title, "Kids Smart Watch")
        self.assertGreater(evidence.confidence_delta, 0.0)
        self.assertEqual(evidence.model_confidence, 0.86)


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_outputs(self) -> None:
        photo_dataset = {
            "receipts": [
                {
                    "merchant": "Barnes & Noble Booksellers #2211",
                    "receipt_urls": ["https://example.com/receipt.jpg"],
                    "items": [
                        {
                            "item_name": "Wavelength by Alex Hague, Justin Vickers",
                            "item_id": "0860001981704",
                            "item_price": "$35.99",
                            "reference_photo_urls": ["https://example.com/doc.jpg"],
                        }
                    ],
                }
            ]
        }
        unanchored_dataset = {
            "receipts": [
                {
                    "merchant": "Goodwill of Central & Northern Arizona",
                    "receipt_urls": ["https://example.com/receipt.jpg"],
                    "items": [
                        {
                            "item_name": "Books",
                            "item_id": "2011857002992",
                            "item_price": "$2.99",
                        }
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            photo_path = temp_path / "photo.json"
            unanchored_path = temp_path / "unanchored.json"
            output_path = temp_path / "output"

            photo_path.write_text(json.dumps(photo_dataset))
            unanchored_path.write_text(json.dumps(unanchored_dataset))

            results = run_pipeline(
                photo_anchored_path=str(photo_path),
                unanchored_path=str(unanchored_path),
                output_dir=output_path,
                photo_analyzer=NoopPhotoAnalyzer(),
                adjudicator=NoopEvidenceAdjudicator(),
                shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
                sibling_context_adjudicator=NoopSiblingContextAdjudicator(),
            )

            self.assertIn("photo_anchored", results)
            self.assertTrue((output_path / "photo_anchored_predictions.json").exists())
            self.assertTrue((output_path / "unanchored_predictions.json").exists())
            self.assertTrue((output_path / "summary.md").exists())
            self.assertTrue((output_path / "index.html").exists())
            self.assertTrue((output_path / "latency_report.json").exists())
            latency_report = json.loads((output_path / "latency_report.json").read_text())
            self.assertIn("aggregate_item_latency", latency_report)
            self.assertGreaterEqual(results["photo_anchored"][0].latency_metrics.total_item_ms, 0.0)

    def test_local_only_mode_disables_expensive_paths(self) -> None:
        photo_dataset = {
            "receipts": [
                {
                    "merchant": "Walmart",
                    "items": [
                        {
                            "item_name": "ABBR X",
                            "item_id": "123456789012",
                            "item_price": "$5.99",
                            "reference_photo_urls": ["https://example.com/photo.jpg"],
                        }
                    ],
                }
            ]
        }
        unanchored_dataset = {"receipts": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            photo_path = temp_path / "photo.json"
            unanchored_path = temp_path / "unanchored.json"
            photo_path.write_text(json.dumps(photo_dataset))
            unanchored_path.write_text(json.dumps(unanchored_dataset))
            analyzer = FakePhotoAnalyzer(
                PhotoEvidence(
                    used=True,
                    success=True,
                    summary="Should not run.",
                    suggested_title="Should Not Run",
                    analyzer_name="fake",
                )
            )
            results = run_pipeline(
                photo_anchored_path=str(photo_path),
                unanchored_path=str(unanchored_path),
                output_dir=temp_path / "output",
                photo_analyzer=analyzer,
                adjudicator=NoopEvidenceAdjudicator(),
                shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
                sibling_context_adjudicator=NoopSiblingContextAdjudicator(),
                pipeline_mode="local_only",
            )
            self.assertEqual(analyzer.calls, 0)
            self.assertFalse(results["photo_anchored"][0].photo_analysis_attempted)

    def test_photo_cache_reuses_same_url(self) -> None:
        photo_dataset = {
            "receipts": [
                {
                    "merchant": "Walmart",
                    "items": [
                        {
                            "item_name": "ABBR X",
                            "item_id": "123456789012",
                            "item_price": "$5.99",
                            "reference_photo_urls": ["https://example.com/shared.jpg"],
                        },
                        {
                            "item_name": "ABBR Y",
                            "item_id": "123456789013",
                            "item_price": "$6.99",
                            "reference_photo_urls": ["https://example.com/shared.jpg"],
                        },
                    ],
                }
            ]
        }
        unanchored_dataset = {"receipts": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            photo_path = temp_path / "photo.json"
            unanchored_path = temp_path / "unanchored.json"
            output_path = temp_path / "output"
            photo_path.write_text(json.dumps(photo_dataset))
            unanchored_path.write_text(json.dumps(unanchored_dataset))
            analyzer = FakePhotoAnalyzer(
                PhotoEvidence(
                    used=False,
                    success=True,
                    summary="Specific product visible.",
                    extracted_signals=["visible title"],
                    model_confidence=0.8,
                    model_suggested_title="Specific Shared Product",
                    analyzer_name="fake",
                )
            )
            run_pipeline(
                photo_anchored_path=str(photo_path),
                unanchored_path=str(unanchored_path),
                output_dir=output_path,
                photo_analyzer=analyzer,
                adjudicator=NoopEvidenceAdjudicator(),
                shared_photo_analyzer=NoopSharedPhotoAnalyzer(),
                sibling_context_adjudicator=NoopSiblingContextAdjudicator(),
                pipeline_mode="photo_ai",
                enable_cache=True,
            )
            report = json.loads((output_path / "latency_report.json").read_text())
            self.assertEqual(analyzer.calls, 1)
            self.assertEqual(report["cache_stats"]["photo_cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
