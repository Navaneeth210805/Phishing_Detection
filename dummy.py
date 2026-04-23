from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from datasets import load_dataset
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from dom_tree_builder import DOMTreeBuilder


INTERACTIVE_TAGS = {"form", "input", "button", "a", "iframe", "select", "textarea"}
GENERIC_CONTAINER_TAGS = {"div", "span", "section", "article", "main", "header", "footer", "nav"}

SUSPICIOUS_TEXT_KEYWORDS = {
    "login",
    "sign in",
    "verify",
    "password",
    "otp",
    "secure",
    "update",
    "bank",
    "account",
}

NOISY_ATTR_PREFIXES = ("data-", "aria-")
NOISY_ATTR_EXACT = {"id", "class", "style", "nonce", "integrity", "crossorigin"}


@dataclass
class DOMNodeInfo:
    """Enhanced DOM node with attribute information."""
    
    node_id: int
    tag: str
    depth: int
    role: str
    attributes: Dict[str, str]
    attr_signature: Tuple[str, ...]
    text_content: str
    child_ids: List[int]
    parent_id: Optional[int]
    
    def get_attribute_vector(self) -> Dict[str, str]:
        """Return normalized attribute vector for comparison."""
        return {k: v for k, v in self.attributes.items() 
                if k not in NOISY_ATTR_EXACT and not k.startswith(NOISY_ATTR_PREFIXES)}


@dataclass
class DOMProfile:
    """Canonical semantic profile extracted from one HTML document."""

    role_counts: Counter
    path_ngrams: Counter
    depth_hist: Counter
    behavior_edges: Counter
    text_keywords: Counter
    metrics: Dict[str, float]
    node_list: List[DOMNodeInfo] = field(default_factory=list)
    attribute_features: Counter = field(default_factory=Counter)

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "role_counts": dict(self.role_counts),
            "path_ngrams": dict(self.path_ngrams),
            "depth_hist": dict(self.depth_hist),
            "behavior_edges": dict(self.behavior_edges),
            "text_keywords": dict(self.text_keywords),
            "metrics": self.metrics,
            "attribute_features": dict(self.attribute_features),
        }


def _extract_domain(host_or_url: str) -> str:
    """Return lowercase host/domain from URL or host string."""
    if not host_or_url:
        return ""
    if "://" in host_or_url:
        parsed = urlparse(host_or_url)
        return (parsed.hostname or "").lower()
    return host_or_url.lower()


def _is_hidden_attrs(attrs: Dict[str, str]) -> bool:
    """Visibility normalization across common hiding mechanisms."""
    if "hidden" in attrs:
        return True

    aria_hidden = str(attrs.get("aria-hidden", "")).strip().lower()
    if aria_hidden in {"true", "1"}:
        return True

    style = str(attrs.get("style", "")).lower().replace(" ", "")
    if not style:
        return False

    hidden_patterns = [
        "display:none",
        "visibility:hidden",
        "opacity:0",
        "height:0",
        "width:0",
    ]
    return any(p in style for p in hidden_patterns)


def _classify_url(url_value: str, base_domain: str) -> str:
    """Classify URL target type in a robust coarse bucket."""
    value = (url_value or "").strip().lower()
    if value == "":
        return "empty"
    if value.startswith("#"):
        return "anchor"
    if value.startswith("javascript:"):
        return "javascript"
    if value.startswith("data:"):
        return "data"

    try:
        parsed = urlparse(value)
    except ValueError:
        # Malformed URLs (for example invalid IPv6 literals) should not abort
        # large streaming extraction jobs.
        return "malformed"
    if parsed.scheme in {"http", "https"}:
        target = (parsed.hostname or "").lower()
        if not target:
            return "relative"
        if base_domain and target == base_domain:
            return "internal"
        if base_domain and target.endswith("." + base_domain):
            return "internal"
        return "external"

    if value.startswith("/") or value.startswith("./") or value.startswith("../"):
        return "relative"

    return "relative"


def _input_role(attrs: Dict[str, str]) -> str:
    t = str(attrs.get("type", "text")).strip().lower()
    mapping = {
        "password": "input_password",
        "email": "input_email",
        "hidden": "input_hidden",
        "submit": "input_submit",
        "button": "input_button",
        "checkbox": "input_checkbox",
        "radio": "input_radio",
        "tel": "input_tel",
    }
    return mapping.get(t, "input_text")


def _tag_role(tag_name: str, attrs: Dict[str, str], base_domain: str) -> str:
    """Map tag to semantic role bucket."""
    name = (tag_name or "").lower()

    if name == "input":
        return _input_role(attrs)
    if name == "form":
        action_class = _classify_url(str(attrs.get("action", "")), base_domain)
        return "form_" + action_class
    if name == "a":
        href_class = _classify_url(str(attrs.get("href", "")), base_domain)
        return "anchor_" + href_class
    if name == "iframe":
        src_class = _classify_url(str(attrs.get("src", "")), base_domain)
        return "iframe_" + src_class
    if name == "button":
        return "button"
    if name == "script":
        return "script"

    if name in GENERIC_CONTAINER_TAGS:
        return "container"
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return "heading"
    if name in {"p", "label", "small", "strong"}:
        return "text"

    return name


def _normalized_attr_keys(attrs: Dict[str, str]) -> Tuple[str, ...]:
    """Use only stable attribute key signatures."""
    keys = []
    for key in attrs.keys():
        k = str(key).lower()
        if k in NOISY_ATTR_EXACT:
            continue
        if k.startswith(NOISY_ATTR_PREFIXES):
            continue
        keys.append(k)
    keys.sort()
    return tuple(keys)


def _counter_jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    keys = set(a.keys()) | set(b.keys())
    inter = 0.0
    union = 0.0
    for k in keys:
        av = float(a.get(k, 0))
        bv = float(b.get(k, 0))
        inter += min(av, bv)
        union += max(av, bv)
    return inter / union if union > 0 else 0.0


def _attribute_similarity(attrs1: Dict[str, str], attrs2: Dict[str, str]) -> float:
    """Compare two attribute dictionaries using Jaccard similarity."""
    if not attrs1 and not attrs2:
        return 1.0
    if not attrs1 or not attrs2:
        return 0.0
    
    keys1 = set(attrs1.keys())
    keys2 = set(attrs2.keys())
    
    # Key overlap
    key_intersection = keys1 & keys2
    key_union = keys1 | keys2
    
    if not key_union:
        return 0.0
    
    key_similarity = len(key_intersection) / len(key_union)
    
    # Value similarity for common keys
    value_matches = 0
    for key in key_intersection:
        val1 = str(attrs1.get(key, "")).lower().strip()
        val2 = str(attrs2.get(key, "")).lower().strip()
        if val1 == val2:
            value_matches += 1
    
    value_similarity = value_matches / len(key_intersection) if key_intersection else 0.0
    
    # Combined attribute similarity: 60% key structure, 40% value matching
    return 0.6 * key_similarity + 0.4 * value_similarity


def _tag_attribute_signature(tag: str, attrs: Dict[str, str]) -> str:
    """Create a comprehensive tag+attribute signature for matching."""
    clean_attrs = _normalized_attr_keys(attrs)
    sig = f"{tag}|" + ",".join(clean_attrs) if clean_attrs else tag
    return sig


class DOMOnlyRobustDetector:
    """
    DOM-only detector designed for resilience against syntax-level HTML rewrites.
    Now includes attribute-based comparison with parent-to-child tree traversal.
    """

    def __init__(self, max_depth: int = 12, ngram_size: int = 3):
        self.max_depth = max_depth
        self.ngram_size = ngram_size

    def build_profile(self, html: str, page_domain: str = "") -> DOMProfile:
        builder = DOMTreeBuilder()
        builder.build_from_html_string(html or "")
        domain = _extract_domain(page_domain)

        role_counts: Counter = Counter()
        path_ngrams: Counter = Counter()
        depth_hist: Counter = Counter()
        behavior_edges: Counter = Counter()
        text_keywords: Counter = Counter()
        attribute_features: Counter = Counter()
        node_list: List[DOMNodeInfo] = []

        if not builder.dom_array:
            return DOMProfile(
                role_counts=role_counts,
                path_ngrams=path_ngrams,
                depth_hist=depth_hist,
                behavior_edges=behavior_edges,
                text_keywords=text_keywords,
                metrics={
                    "total_nodes": 0.0,
                    "interactive_ratio": 0.0,
                    "edge_count": 0.0,
                    "keyword_density": 0.0,
                },
                node_list=node_list,
                attribute_features=attribute_features,
            )

        id_to_node = {n.id: n for n in builder.dom_array}

        root_id = 0
        for n in builder.dom_array:
            if n.parent_id is None:
                root_id = n.id
                break

        def walk(node_id: int, depth: int, role_path: List[str]) -> None:
            if depth > self.max_depth:
                return
            node = id_to_node.get(node_id)
            if node is None:
                return
            attrs = node.attributes or {}
            if _is_hidden_attrs(attrs):
                return

            role = _tag_role(node.tag, attrs, domain)
            attr_sig = _normalized_attr_keys(attrs)
            attr_vector = {k: v for k, v in attrs.items() 
                          if k not in NOISY_ATTR_EXACT and not k.startswith(NOISY_ATTR_PREFIXES)}

            # Create enhanced node info
            node_info = DOMNodeInfo(
                node_id=node.id,
                tag=node.tag,
                depth=depth,
                role=role,
                attributes=attr_vector,
                attr_signature=attr_sig,
                text_content=node.text_content or "",
                child_ids=node.child_ids or [],
                parent_id=node.parent_id,
            )
            node_list.append(node_info)

            # Record attribute features
            tag_attr_sig = _tag_attribute_signature(node.tag, attrs)
            attribute_features[tag_attr_sig] += 1
            
            for attr_key in attr_sig:
                attribute_features[f"{node.tag}.{attr_key}"] += 1

            # Reduce impact of meaningless wrappers by limiting container explosion.
            child_count = len(node.child_ids or [])
            if role == "container" and child_count <= 1:
                next_path = role_path
            else:
                role_counts[role] += 1
                depth_hist[depth] += 1
                next_path = role_path + [role]

                if len(next_path) >= self.ngram_size:
                    ngram = ">".join(next_path[-self.ngram_size :])
                    path_ngrams[ngram] += 1

                if attr_sig:
                    path_ngrams[role + "|attrs=" + ",".join(attr_sig)] += 1

            self._extract_behavior(node, id_to_node, domain, behavior_edges)
            self._extract_text_keywords(node.text_content, text_keywords)

            for child_id in node.child_ids or []:
                walk(child_id, depth + 1, next_path)

        walk(root_id, 0, [])

        total_nodes = float(sum(role_counts.values()))
        interactive_nodes = float(sum(role_counts[r] for r in role_counts if r.split("_")[0] in INTERACTIVE_TAGS or r.startswith("input")))

        metrics = {
            "total_nodes": total_nodes,
            "interactive_ratio": (interactive_nodes / total_nodes) if total_nodes > 0 else 0.0,
            "edge_count": float(sum(behavior_edges.values())),
            "keyword_density": (sum(text_keywords.values()) / max(total_nodes, 1.0)),
        }

        return DOMProfile(
            role_counts=role_counts,
            path_ngrams=path_ngrams,
            depth_hist=depth_hist,
            behavior_edges=behavior_edges,
            text_keywords=text_keywords,
            metrics=metrics,
            node_list=node_list,
            attribute_features=attribute_features,
        )

    def _extract_behavior(self, node, id_to_node: Dict[int, object], domain: str, edges: Counter) -> None:
        name = (node.tag or "").lower()
        attrs = node.attributes or {}

        if name == "form":
            action_class = _classify_url(str(attrs.get("action", "")), domain)
            method = str(attrs.get("method", "get")).strip().lower()
            has_password = self._subtree_has_password(node.id, id_to_node)
            edges[f"form_submit|method={method}|target={action_class}|pwd={int(has_password)}"] += 1

        elif name == "a":
            href_class = _classify_url(str(attrs.get("href", "")), domain)
            edges[f"anchor_nav|target={href_class}"] += 1

        elif name == "iframe":
            src_class = _classify_url(str(attrs.get("src", "")), domain)
            edges[f"iframe_load|target={src_class}"] += 1

        elif name == "script":
            script_text = self._subtree_text(node.id, id_to_node).lower()
            patterns = {
                "js_redirect": r"window\.location|location\.href|location\.replace",
                "js_popup": r"window\.open\(|prompt\(|alert\(",
                "js_form_submit": r"\.submit\(",
                "js_network": r"fetch\(|xmlhttprequest|axios\.",
                "js_base64": r"atob\(|fromcharcode\(|eval\(",
            }
            for key, pattern in patterns.items():
                if re.search(pattern, script_text):
                    edges[key] += 1

    def _extract_text_keywords(self, text: str, text_keywords: Counter) -> None:
        text = (text or "").lower()
        if not text:
            return

        for kw in SUSPICIOUS_TEXT_KEYWORDS:
            if kw in text:
                text_keywords[kw] += 1

    def _subtree_has_password(self, root_id: int, id_to_node: Dict[int, object]) -> bool:
        stack = [root_id]
        while stack:
            nid = stack.pop()
            n = id_to_node.get(nid)
            if n is None:
                continue
            if (n.tag or "").lower() == "input":
                t = str((n.attributes or {}).get("type", "")).strip().lower()
                if t == "password":
                    return True
            stack.extend(n.child_ids or [])
        return False

    def _subtree_text(self, root_id: int, id_to_node: Dict[int, object]) -> str:
        parts: List[str] = []
        stack = [root_id]
        while stack:
            nid = stack.pop()
            n = id_to_node.get(nid)
            if n is None:
                continue
            if n.text_content:
                parts.append(n.text_content)
            stack.extend(n.child_ids or [])
        return " ".join(parts)

    def _compare_node_pair(self, node1: DOMNodeInfo, node2: DOMNodeInfo) -> float:
        """
        Compare two individual nodes based on tag, role, and attributes.
        Returns a similarity score between 0 and 1.
        """
        # Tag match (40% weight)
        tag_match = 1.0 if node1.tag == node2.tag else 0.0
        
        # Role match (30% weight)
        role_match = 1.0 if node1.role == node2.role else 0.3 if node1.role.split("_")[0] == node2.role.split("_")[0] else 0.0
        
        # Attribute similarity (30% weight)
        attr_sim = _attribute_similarity(node1.attributes, node2.attributes)
        
        node_similarity = 0.4 * tag_match + 0.3 * role_match + 0.3 * attr_sim
        return node_similarity

    def _hungarian_maximize(self, score_matrix: List[List[float]]) -> List[float]:
        """
        Solve one-to-one assignment that maximizes total score.
        Returns best matched score for each row in score_matrix.
        """
        if not score_matrix or not score_matrix[0]:
            return []

        n_rows = len(score_matrix)
        n_cols = len(score_matrix[0])

        # Ensure columns >= rows by padding with dummy zero-score columns.
        if n_cols < n_rows:
            pad = n_rows - n_cols
            for r in range(n_rows):
                score_matrix[r].extend([0.0] * pad)
            n_cols = n_rows

        # Hungarian algorithm for min-cost assignment on rectangular matrix (n_rows <= n_cols).
        # Convert max score to min cost.
        max_score = 0.0
        for row in score_matrix:
            for v in row:
                if v > max_score:
                    max_score = v

        cost = [[max_score - v for v in row] for row in score_matrix]

        u = [0.0] * (n_rows + 1)
        v = [0.0] * (n_cols + 1)
        p = [0] * (n_cols + 1)
        way = [0] * (n_cols + 1)

        for i in range(1, n_rows + 1):
            p[0] = i
            j0 = 0
            minv = [float("inf")] * (n_cols + 1)
            used = [False] * (n_cols + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0
                for j in range(1, n_cols + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(0, n_cols + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = [-1] * n_rows
        for j in range(1, n_cols + 1):
            if p[j] > 0 and p[j] <= n_rows:
                assignment[p[j] - 1] = j - 1

        matched_scores: List[float] = []
        for i, j in enumerate(assignment):
            if j < 0 or j >= len(score_matrix[i]):
                matched_scores.append(0.0)
            else:
                matched_scores.append(score_matrix[i][j])
        return matched_scores

    def _tree_traversal_comparison(self, ref_profile: DOMProfile, cand_profile: DOMProfile) -> Tuple[float, Dict[str, float]]:
        """
        Perform parent-to-child tree traversal comparison (n×n matching).
        Matches nodes from reference tree to candidate tree and computes overall similarity.
        
        Returns: (overall_similarity, detailed_scores)
        """
        if not ref_profile.node_list or not cand_profile.node_list:
            return 0.0, {}

        # Context features are not separate checks. They are blended into traversal scoring
        # to make parent-to-child matching more robust under noisy rewrites.
        path_ngram_context = _counter_jaccard(ref_profile.path_ngrams, cand_profile.path_ngrams)
        attribute_context = _counter_jaccard(ref_profile.attribute_features, cand_profile.attribute_features)
        behavior_context = _counter_jaccard(ref_profile.behavior_edges, cand_profile.behavior_edges)
        text_context = _counter_jaccard(ref_profile.text_keywords, cand_profile.text_keywords)
        depth_context = _counter_jaccard(ref_profile.depth_hist, cand_profile.depth_hist)

        global_context = (
            0.35 * path_ngram_context +
            0.35 * attribute_context +
            0.15 * behavior_context +
            0.10 * depth_context +
            0.05 * text_context
        )
        
        # Build parent-child relationships for both profiles
        ref_nodes_by_depth = {}
        cand_nodes_by_depth = {}
        
        for node in ref_profile.node_list:
            depth = node.depth
            if depth not in ref_nodes_by_depth:
                ref_nodes_by_depth[depth] = []
            ref_nodes_by_depth[depth].append(node)
        
        for node in cand_profile.node_list:
            depth = node.depth
            if depth not in cand_nodes_by_depth:
                cand_nodes_by_depth[depth] = []
            cand_nodes_by_depth[depth].append(node)
        
        ref_id_to_node = {n.node_id: n for n in ref_profile.node_list}
        cand_id_to_node = {n.node_id: n for n in cand_profile.node_list}

        # Score matching for each depth level (parent to child traversal)
        depth_scores = []
        node_match_count = 0
        total_comparisons = 0
        
        for depth in sorted(set(ref_nodes_by_depth.keys()) | set(cand_nodes_by_depth.keys())):
            ref_depth_nodes = ref_nodes_by_depth.get(depth, [])
            cand_depth_nodes = cand_nodes_by_depth.get(depth, [])
            
            if not ref_depth_nodes or not cand_depth_nodes:
                continue
            
            # One-to-one matching per depth via Hungarian assignment.
            sim_matrix: List[List[float]] = []
            for ref_node in ref_depth_nodes:
                row: List[float] = []
                for cand_node in cand_depth_nodes:
                    base_sim = self._compare_node_pair(ref_node, cand_node)

                    parent_consistency = 0.0
                    if ref_node.parent_id is not None and cand_node.parent_id is not None:
                        ref_parent = ref_id_to_node.get(ref_node.parent_id)
                        cand_parent = cand_id_to_node.get(cand_node.parent_id)
                        if ref_parent is not None and cand_parent is not None:
                            if ref_parent.role == cand_parent.role:
                                parent_consistency = 0.10
                            elif ref_parent.role.split("_")[0] == cand_parent.role.split("_")[0]:
                                parent_consistency = 0.05

                    sim = min(1.0, 0.82 * base_sim + 0.10 * global_context + parent_consistency)
                    row.append(sim)
                sim_matrix.append(row)

            best_matches_per_ref = self._hungarian_maximize(sim_matrix)

            for matched_score in best_matches_per_ref:
                total_comparisons += 1
                if matched_score >= 0.7:
                    node_match_count += 1
            
            if best_matches_per_ref:
                avg_depth_score = sum(best_matches_per_ref) / len(best_matches_per_ref)
                depth_scores.append(avg_depth_score)
        
        # Final score is still tree traversal similarity, context only stabilizes matching.
        if depth_scores:
            raw_tree_similarity = sum(depth_scores) / len(depth_scores)
        else:
            raw_tree_similarity = 0.0

        tree_similarity = 0.90 * raw_tree_similarity + 0.10 * global_context
        
        match_ratio = node_match_count / total_comparisons if total_comparisons > 0 else 0.0
        
        detailed_scores = {
            "tree_traversal_similarity": round(tree_similarity, 6),
            "raw_tree_traversal_similarity": round(raw_tree_similarity, 6),
            "node_match_ratio": round(match_ratio, 6),
            "total_comparisons": float(total_comparisons),
            "successful_matches": float(node_match_count),
            "context_path_ngram": round(path_ngram_context, 6),
            "context_attributes": round(attribute_context, 6),
            "context_behavior": round(behavior_context, 6),
            "context_depth": round(depth_context, 6),
            "context_text": round(text_context, 6),
            "context_blend": round(global_context, 6),
        }
        
        return tree_similarity, detailed_scores

    def compare_profiles(self, ref: DOMProfile, cand: DOMProfile) -> Dict[str, float]:
        """Return diagnostics where final similarity is a single tree-traversal score."""
        tree_traversal_sim, tree_scores = self._tree_traversal_comparison(ref, cand)
        dom_semantic_similarity = tree_traversal_sim

        # Critical risk indicator: password collection with non-internal action.
        external_pwd_forms = 0
        for edge, count in cand.behavior_edges.items():
            if edge.startswith("form_submit") and "pwd=1" in edge and ("target=external" in edge or "target=empty" in edge):
                external_pwd_forms += int(count)

        result = {
            "tree_traversal_similarity": round(tree_traversal_sim, 6),
            "dom_semantic_similarity": round(dom_semantic_similarity, 6),
            "external_password_form_count": float(external_pwd_forms),
        }
        
        # Add tree traversal details
        result.update(tree_scores)
        
        return result

    def compare_html(self, reference_html: str, candidate_html: str, reference_domain: str = "", candidate_domain: str = "") -> Dict[str, object]:
        ref_profile = self.build_profile(reference_html, reference_domain)
        cand_profile = self.build_profile(candidate_html, candidate_domain)
        sims = self.compare_profiles(ref_profile, cand_profile)

        sem_sim = float(sims["dom_semantic_similarity"])
        external_pwd = int(sims["external_password_form_count"])

        if sem_sim >= 0.85:
            verdict = "high_semantic_match"
        elif sem_sim >= 0.70:
            verdict = "moderate_semantic_match"
        else:
            verdict = "low_semantic_match"

        if external_pwd > 0:
            risk = "high_risk_behavior"
        elif sem_sim >= 0.85:
            risk = "possible_brand_mimic"
        elif sem_sim >= 0.70:
            risk = "needs_manual_review"
        else:
            risk = "low_structural_match"

        return {
            "verdict": verdict,
            "risk_label": risk,
            "scores": sims,
            "reference_metrics": ref_profile.metrics,
            "candidate_metrics": cand_profile.metrics,
        }

    def detect_against_references(self, candidate_html: str, reference_html_map: Dict[str, str], candidate_domain: str = "") -> Dict[str, object]:
        """Compare candidate DOM against multiple reference DOM pages."""
        if not reference_html_map:
            raise ValueError("reference_html_map must not be empty")

        results = []
        for ref_name, ref_html in reference_html_map.items():
            result = self.compare_html(ref_html, candidate_html, reference_domain=ref_name, candidate_domain=candidate_domain)
            results.append((ref_name, result))

        best_name, best_result = max(results, key=lambda x: x[1]["scores"]["dom_semantic_similarity"])
        return {
            "best_reference": best_name,
            "best_result": best_result,
            "all_results": {name: res for name, res in results},
        }


class HTMLUnsupervisedDOMClassifier:
    """HTML-only one-class phishing detector based on DOM semantic profiles."""

    def __init__(self, hashed_dim: int = 4096, contamination: float = 0.15, random_state: int = 42):
        self.hashed_dim = hashed_dim
        self.contamination = contamination
        self.random_state = random_state

        self.dom_detector = DOMOnlyRobustDetector()
        self.hasher = FeatureHasher(n_features=hashed_dim, input_type="dict", alternate_sign=False)
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    @staticmethod
    def _normalize_label(raw) -> int:
        try:
            if isinstance(raw, str):
                return 1 if raw.lower() in {"phish", "phishing", "1"} else 0
            return 1 if int(raw) == 1 else 0
        except Exception:
            return 0

    def _flatten_profile(self, profile: DOMProfile) -> Dict[str, float]:
        feat: Dict[str, float] = {}

        for k, v in profile.metrics.items():
            feat[f"metric::{k}"] = float(v)

        groups = [
            ("role", profile.role_counts),
            ("path", profile.path_ngrams),
            ("depth", profile.depth_hist),
            ("behavior", profile.behavior_edges),
            ("text", profile.text_keywords),
            ("attr", profile.attribute_features),
        ]
        for prefix, counter in groups:
            for k, v in counter.items():
                feat[f"{prefix}::{k}"] = float(v)

        return feat

    def extract_html_feature_dict(self, html: str) -> Dict[str, float]:
        try:
            profile = self.dom_detector.build_profile(html or "", page_domain="")
            return self._flatten_profile(profile)
        except Exception:
            # Keep extraction resilient to rare malformed HTML/attributes.
            return {}

    def extract_html_feature_matrix(self, html_batch: List[str]) -> np.ndarray:
        dict_batch = [self.extract_html_feature_dict(h) for h in html_batch]
        return self.hasher.transform(dict_batch).toarray().astype(np.float32)

    def stream_split_to_feature_store(
        self,
        split_name: str,
        feature_dir: str,
        batch_size: int = 512,
        hf_token: Optional[str] = None,
        resume: bool = False,
    ) -> List[str]:
        """
        Stream split from phreshphish and save only compact DOM feature chunks.
        This does not persist raw HTML or URL fields.
        """
        out_dir = Path(feature_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        chunk_pattern = f"{split_name}_domfeat_chunk_*.npz"
        existing_chunks = sorted(out_dir.glob(chunk_pattern))
        chunk_idx = 0
        seen = 0
        if resume and existing_chunks:
            chunk_idx = len(existing_chunks)
            seen = chunk_idx * batch_size
            print(
                f"[unsup-extract] resume enabled: found={len(existing_chunks)} skip_samples={seen:,} next_chunk={chunk_idx}",
                flush=True,
            )

        print(
            f"[unsup-extract] split={split_name} batch_size={batch_size}",
            flush=True,
        )
        dataset = load_dataset("phreshphish/phreshphish", split=split_name, streaming=True, token=hf_token)
        if seen > 0:
            if hasattr(dataset, "skip"):
                dataset = dataset.skip(seen)
            else:
                # Fallback for iterables without efficient skip support.
                iterator = iter(dataset)
                for _ in range(seen):
                    try:
                        next(iterator)
                    except StopIteration:
                        break
                dataset = iterator
        html_batch: List[str] = []
        label_batch: List[int] = []
        saved_files: List[str] = [str(p) for p in existing_chunks] if resume else []

        for sample in dataset:
            html_batch.append(sample.get("html", "") or "")
            label_batch.append(self._normalize_label(sample.get("label", 0)))
            seen += 1

            if len(html_batch) >= batch_size:
                X = self.extract_html_feature_matrix(html_batch)
                y = np.array(label_batch, dtype=np.int8)
                out_path = out_dir / f"{split_name}_domfeat_chunk_{chunk_idx:05d}.npz"
                np.savez_compressed(out_path, X=X, y=y)
                saved_files.append(str(out_path))
                html_batch = []
                label_batch = []
                chunk_idx += 1

                if chunk_idx % 5 == 0:
                    print(
                        f"[unsup-extract] processed={seen:,} saved_chunks={chunk_idx} last={out_path.name}",
                        flush=True,
                    )

        if html_batch:
            X = self.extract_html_feature_matrix(html_batch)
            y = np.array(label_batch, dtype=np.int8)
            out_path = out_dir / f"{split_name}_domfeat_chunk_{chunk_idx:05d}.npz"
            np.savez_compressed(out_path, X=X, y=y)
            saved_files.append(str(out_path))

        print(
            f"[unsup-extract] completed processed={seen:,} total_chunks={len(saved_files)}",
            flush=True,
        )

        return saved_files

    def _load_chunks(self, chunk_files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []

        total_chunks = len(chunk_files)
        for idx, path in enumerate(chunk_files, start=1):
            with np.load(path) as data:
                x_parts.append(data["X"])
                y_parts.append(data["y"])

            if idx % 200 == 0 or idx == total_chunks:
                print(
                    f"[unsup-train] loaded_chunks={idx}/{total_chunks}",
                    flush=True,
                )

        if not x_parts:
            return (
                np.empty((0, self.hashed_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int8),
            )

        return np.vstack(x_parts).astype(np.float32), np.concatenate(y_parts).astype(np.int8)

    def fit_from_feature_store_streaming(self, train_chunk_files: List[str]) -> Dict[str, int]:
        """Fit incrementally from chunks to avoid loading all data at once."""
        print(
            f"[unsup-train] starting streaming fit chunks={len(train_chunk_files)} hashed_dim={self.hashed_dim}",
            flush=True,
        )

        buffer_size = 100000  # Accumulate benign samples
        fit_size = 50000      # Only fit model on 50k benign (fits safely in memory)
        X_benign_buffer: List[np.ndarray] = []
        buffered_benign = 0
        total_benign = 0
        total_samples = 0
        first_fit = True

        total_chunks = len(train_chunk_files)
        for chunk_idx, path in enumerate(train_chunk_files, start=1):
            with np.load(path) as data:
                X = data["X"].astype(np.float32)
                y = data["y"].astype(np.int8)

            total_samples += len(X)
            benign_mask = y == 0
            X_benign_chunk = X[benign_mask]

            if len(X_benign_chunk) > 0:
                X_benign_buffer.append(X_benign_chunk)
                buffered_benign += len(X_benign_chunk)
                total_benign += len(X_benign_chunk)

            # Progress every 200 chunks
            if chunk_idx % 200 == 0 or chunk_idx == total_chunks:
                print(
                    f"[unsup-train] loaded_chunks={chunk_idx}/{total_chunks} buffered={buffered_benign:,} total_benign={total_benign:,}",
                    flush=True,
                )

            # When buffer reaches size, fit on subset and reset
            if buffered_benign >= buffer_size or chunk_idx == total_chunks:
                if X_benign_buffer:
                    X_benign_combined = np.vstack(X_benign_buffer)
                    
                    # Subsample to fit_size to avoid OOM
                    if len(X_benign_combined) > fit_size:
                        indices = np.random.choice(len(X_benign_combined), fit_size, replace=False)
                        X_benign_fit = X_benign_combined[indices]
                    else:
                        X_benign_fit = X_benign_combined

                    print(
                        f"[unsup-train] fitting on benign={len(X_benign_fit):,}",
                        flush=True,
                    )

                    if first_fit:
                        print("[unsup-train] fitting scaler", flush=True)
                        self.scaler.fit(X_benign_fit)
                        print("[unsup-train] transforming and fitting IsolationForest", flush=True)
                        X_benign_scaled = self.scaler.transform(X_benign_fit)
                        self.model.fit(X_benign_scaled)
                        self.is_fitted = True
                        first_fit = False
                    else:
                        # Update and refit
                        print("[unsup-train] updating scaler and refitting model", flush=True)
                        self.scaler.partial_fit(X_benign_fit)
                        X_benign_scaled = self.scaler.transform(X_benign_fit)
                        self.model.fit(X_benign_scaled)

                    X_benign_buffer = []
                    buffered_benign = 0

        if not self.is_fitted:
            raise ValueError("No benign samples found to fit model.")

        print("[unsup-train] fit complete", flush=True)
        return {
            "train_total": int(total_samples),
            "train_benign_total_seen": int(total_benign),
        }


    def fit_from_feature_store(self, train_chunk_files: List[str]) -> Dict[str, int]:
        print(
            f"[unsup-train] starting with chunks={len(train_chunk_files)} hashed_dim={self.hashed_dim}",
            flush=True,
        )
        X_train, y_train = self._load_chunks(train_chunk_files)
        if len(X_train) == 0:
            raise ValueError("No training features found.")

        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        if len(X_benign) == 0:
            raise ValueError("No benign samples available to fit one-class model.")

        print(
            f"[unsup-train] loaded benign={len(X_benign):,} total={len(X_train):,}",
            flush=True,
        )

        # Subsample to 200k benign samples to avoid OOM on large datasets.
        max_samples = 200000
        if len(X_benign) > max_samples:
            print(
                f"[unsup-train] subsampling benign {len(X_benign):,} -> {max_samples:,}",
                flush=True,
            )
            indices = np.random.choice(len(X_benign), max_samples, replace=False)
            X_benign = X_benign[indices]

        print(
            f"[unsup-train] fitting scaler on benign={len(X_benign):,}",
            flush=True,
        )
        self.scaler.fit(X_benign)

        print("[unsup-train] transforming benign features", flush=True)
        X_benign_scaled = self.scaler.transform(X_benign)

        print("[unsup-train] fitting IsolationForest", flush=True)
        self.model.fit(X_benign_scaled)
        self.is_fitted = True

        print("[unsup-train] fit complete", flush=True)
        return {
            "train_total": int(len(X_train)),
            "train_benign_used": int(len(X_benign)),
        }

    def evaluate_from_feature_store(self, test_chunk_files: List[str]) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")

        X_test, y_test = self._load_chunks(test_chunk_files)
        if len(X_test) == 0:
            raise ValueError("No test features found.")

        X_test_scaled = self.scaler.transform(X_test)
        anomaly_scores = -self.model.decision_function(X_test_scaled)
        y_pred = (self.model.predict(X_test_scaled) == -1).astype(np.int8)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "tp": float(np.sum((y_pred == 1) & (y_test == 1))),
            "tn": float(np.sum((y_pred == 0) & (y_test == 0))),
            "fp": float(np.sum((y_pred == 1) & (y_test == 0))),
            "fn": float(np.sum((y_pred == 0) & (y_test == 1))),
        }
        if len(np.unique(y_test)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_test, anomaly_scores))
        else:
            metrics["roc_auc"] = float("nan")
        return metrics

    def predict_html(self, html: str) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        X = self.extract_html_feature_matrix([html])
        X_scaled = self.scaler.transform(X)
        anomaly_score = float(-self.model.decision_function(X_scaled)[0])
        pred_is_anomaly = int(self.model.predict(X_scaled)[0] == -1)
        return {
            "label": "Phishing" if pred_is_anomaly == 1 else "Benign",
            "anomaly_score": anomaly_score,
        }

    def save(self, model_path: str) -> None:
        payload = {
            "hashed_dim": self.hashed_dim,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "scaler": self.scaler,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }
        with Path(model_path).open("wb") as f:
            pickle.dump(payload, f)

    def load(self, model_path: str) -> None:
        with Path(model_path).open("rb") as f:
            payload = pickle.load(f)

        self.hashed_dim = int(payload["hashed_dim"])
        self.contamination = float(payload["contamination"])
        self.random_state = int(payload["random_state"])
        self.scaler = payload["scaler"]
        self.model = payload["model"]
        self.is_fitted = bool(payload["is_fitted"])
        self.hasher = FeatureHasher(n_features=self.hashed_dim, input_type="dict", alternate_sign=False)


class HTMLSupervisedDOMClassifier:
    """HTML-only supervised phishing detector using Random Forest on DOM features."""

    def __init__(self, hashed_dim: int = 4096, n_estimators: int = 200, random_state: int = 42):
        self.hashed_dim = hashed_dim
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.dom_detector = DOMOnlyRobustDetector()
        self.hasher = FeatureHasher(n_features=hashed_dim, input_type="dict", alternate_sign=False)
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=20,
            min_samples_split=10,
            class_weight="balanced",
        )
        self.is_fitted = False

    @staticmethod
    def _normalize_label(raw) -> int:
        try:
            if isinstance(raw, str):
                return 1 if raw.lower() in {"phish", "phishing", "1"} else 0
            return 1 if int(raw) == 1 else 0
        except Exception:
            return 0

    def _flatten_profile(self, profile: DOMProfile) -> Dict[str, float]:
        feat: Dict[str, float] = {}

        for k, v in profile.metrics.items():
            feat[f"metric::{k}"] = float(v)

        groups = [
            ("role", profile.role_counts),
            ("path", profile.path_ngrams),
            ("depth", profile.depth_hist),
            ("behavior", profile.behavior_edges),
            ("text", profile.text_keywords),
            ("attr", profile.attribute_features),
        ]
        for prefix, counter in groups:
            for k, v in counter.items():
                feat[f"{prefix}::{k}"] = float(v)

        return feat

    def extract_html_feature_dict(self, html: str) -> Dict[str, float]:
        try:
            profile = self.dom_detector.build_profile(html or "", page_domain="")
            return self._flatten_profile(profile)
        except Exception:
            return {}

    def extract_html_feature_matrix(self, html_batch: List[str]) -> np.ndarray:
        dict_batch = [self.extract_html_feature_dict(h) for h in html_batch]
        return self.hasher.transform(dict_batch).toarray().astype(np.float32)

    def _load_chunks(self, chunk_files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []

        total_chunks = len(chunk_files)
        for idx, path in enumerate(chunk_files, start=1):
            with np.load(path) as data:
                x_parts.append(data["X"])
                y_parts.append(data["y"])

            if idx % 200 == 0 or idx == total_chunks:
                print(
                    f"[sup-train] loaded_chunks={idx}/{total_chunks}",
                    flush=True,
                )

        if not x_parts:
            return (
                np.empty((0, self.hashed_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int8),
            )

        return np.vstack(x_parts).astype(np.float32), np.concatenate(y_parts).astype(np.int8)

    def fit_from_feature_store_streaming(self, train_chunk_files: List[str]) -> Dict[str, int]:
        """Fit Random Forest incrementally from chunks using warm_start."""
        print(
            f"[sup-train] starting supervised training chunks={len(train_chunk_files)} hashed_dim={self.hashed_dim}",
            flush=True,
        )

        buffer_size = 50000     # Accumulate both benign and phishing
        X_buffer: List[np.ndarray] = []
        y_buffer: List[np.ndarray] = []
        buffered_samples = 0
        total_benign = 0
        total_phishing = 0
        total_samples = 0
        first_fit = True

        total_chunks = len(train_chunk_files)
        for chunk_idx, path in enumerate(train_chunk_files, start=1):
            with np.load(path) as data:
                X = data["X"].astype(np.float32)
                y = data["y"].astype(np.int8)

            total_samples += len(X)
            total_benign += np.sum(y == 0)
            total_phishing += np.sum(y == 1)

            X_buffer.append(X)
            y_buffer.append(y)
            buffered_samples += len(X)

            # Progress every 200 chunks
            if chunk_idx % 200 == 0 or chunk_idx == total_chunks:
                print(
                    f"[sup-train] loaded_chunks={chunk_idx}/{total_chunks} buffered={buffered_samples:,} benign={total_benign:,} phishing={total_phishing:,}",
                    flush=True,
                )

            # When buffer reaches size, fit and reset
            if buffered_samples >= buffer_size or chunk_idx == total_chunks:
                if X_buffer:
                    X_combined = np.vstack(X_buffer).astype(np.float32)
                    y_combined = np.concatenate(y_buffer).astype(np.int8)

                    print(
                        f"[sup-train] fitting on samples={len(X_combined):,} (benign={np.sum(y_combined == 0)}, phishing={np.sum(y_combined == 1)})",
                        flush=True,
                    )

                    if first_fit:
                        print("[sup-train] fitting scaler", flush=True)
                        self.scaler.fit(X_combined)
                        print("[sup-train] transforming and fitting RandomForest", flush=True)
                        X_combined_scaled = self.scaler.transform(X_combined)
                        self.model.fit(X_combined_scaled, y_combined)
                        self.is_fitted = True
                        first_fit = False
                    else:
                        # Update and refit
                        print("[sup-train] updating scaler and refitting model", flush=True)
                        self.scaler.partial_fit(X_combined)
                        X_combined_scaled = self.scaler.transform(X_combined)
                        self.model.fit(X_combined_scaled, y_combined)

                    X_buffer = []
                    y_buffer = []
                    buffered_samples = 0

        if not self.is_fitted:
            raise ValueError("No training samples found to fit model.")

        print("[sup-train] fit complete", flush=True)
        return {
            "train_total": int(total_samples),
            "train_benign": int(total_benign),
            "train_phishing": int(total_phishing),
        }

    def evaluate_from_feature_store(self, test_chunk_files: List[str]) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")

        X_test, y_test = self._load_chunks(test_chunk_files)
        if len(X_test) == 0:
            raise ValueError("No test features found.")

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "tp": float(np.sum((y_pred == 1) & (y_test == 1))),
            "tn": float(np.sum((y_pred == 0) & (y_test == 0))),
            "fp": float(np.sum((y_pred == 1) & (y_test == 0))),
            "fn": float(np.sum((y_pred == 0) & (y_test == 1))),
        }
        if len(np.unique(y_test)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        else:
            metrics["roc_auc"] = float("nan")
        return metrics

    def predict_html(self, html: str) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        X = self.extract_html_feature_matrix([html])
        X_scaled = self.scaler.transform(X)
        pred_label = int(self.model.predict(X_scaled)[0])
        pred_proba = float(self.model.predict_proba(X_scaled)[0, 1])
        return {
            "label": "Phishing" if pred_label == 1 else "Benign",
            "phishing_probability": pred_proba,
        }

    def save(self, model_path: str) -> None:
        payload = {
            "hashed_dim": self.hashed_dim,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "scaler": self.scaler,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }
        with Path(model_path).open("wb") as f:
            pickle.dump(payload, f)

    def load(self, model_path: str) -> None:
        with Path(model_path).open("rb") as f:
            payload = pickle.load(f)

        self.hashed_dim = int(payload["hashed_dim"])
        self.n_estimators = int(payload["n_estimators"])
        self.random_state = int(payload["random_state"])
        self.scaler = payload["scaler"]
        self.model = payload["model"]
        self.is_fitted = bool(payload["is_fitted"])
        self.hasher = FeatureHasher(n_features=self.hashed_dim, input_type="dict", alternate_sign=False)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _load_reference_map(paths: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for p in paths:
        pp = Path(p)
        mapping[pp.stem] = pp.read_text(encoding="utf-8", errors="ignore")
    return mapping


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DOM-only robust phishing comparison")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compare = sub.add_parser("compare", help="Compare one reference HTML vs one candidate HTML")
    p_compare.add_argument("--reference-html", required=True, help="Path to reference HTML")
    p_compare.add_argument("--candidate-html", required=True, help="Path to candidate HTML")
    p_compare.add_argument("--reference-domain", default="", help="Reference domain/URL (optional)")
    p_compare.add_argument("--candidate-domain", default="", help="Candidate domain/URL (optional)")

    p_detect = sub.add_parser("detect", help="Compare candidate HTML against many reference HTML files")
    p_detect.add_argument("--candidate-html", required=True, help="Path to candidate HTML")
    p_detect.add_argument("--candidate-domain", default="", help="Candidate domain/URL (optional)")
    p_detect.add_argument("--references", nargs="+", required=True, help="List of reference HTML files")

    p_profile = sub.add_parser("profile", help="Export canonical DOM profile as JSON")
    p_profile.add_argument("--html", required=True, help="Path to HTML file")
    p_profile.add_argument("--domain", default="", help="Domain/URL for URL-bucket normalization")

    p_unsup_extract = sub.add_parser(
        "unsup-extract",
        help="Stream phreshphish split and store only HTML DOM feature chunks",
    )
    p_unsup_extract.add_argument("--split", choices=["train", "test"], required=True)
    p_unsup_extract.add_argument("--feature-dir", required=True)
    p_unsup_extract.add_argument("--batch-size", type=int, default=512)
    p_unsup_extract.add_argument("--hf-token", default=None)
    p_unsup_extract.add_argument("--hashed-dim", type=int, default=4096)
    p_unsup_extract.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing chunk files in --feature-dir for the same split",
    )

    p_unsup_train = sub.add_parser(
        "unsup-train",
        help="Train HTML-only one-class model from stored train feature chunks",
    )
    p_unsup_train.add_argument("--train-chunks", nargs="+", required=True)
    p_unsup_train.add_argument("--model-path", required=True)
    p_unsup_train.add_argument("--hashed-dim", type=int, default=4096)
    p_unsup_train.add_argument("--contamination", type=float, default=0.15)

    p_unsup_eval = sub.add_parser(
        "unsup-eval",
        help="Evaluate HTML-only one-class model on test feature chunks",
    )
    p_unsup_eval.add_argument("--model-path", required=True)
    p_unsup_eval.add_argument("--test-chunks", nargs="+", required=True)

    p_unsup_predict = sub.add_parser(
        "unsup-predict",
        help="Predict phishing/benign from HTML only using trained one-class model",
    )
    p_unsup_predict.add_argument("--model-path", required=True)
    p_unsup_predict.add_argument("--html", required=True, help="Path to candidate HTML")

    p_sup_train = sub.add_parser(
        "sup-train",
        help="Train supervised Random Forest model from stored train feature chunks (benign + phishing)",
    )
    p_sup_train.add_argument("--train-chunks", nargs="+", required=True)
    p_sup_train.add_argument("--model-path", required=True)
    p_sup_train.add_argument("--hashed-dim", type=int, default=4096)
    p_sup_train.add_argument("--n-estimators", type=int, default=200)

    p_sup_eval = sub.add_parser(
        "sup-eval",
        help="Evaluate supervised Random Forest model on test feature chunks",
    )
    p_sup_eval.add_argument("--model-path", required=True)
    p_sup_eval.add_argument("--test-chunks", nargs="+", required=True)

    p_sup_predict = sub.add_parser(
        "sup-predict",
        help="Predict phishing/benign from HTML using trained supervised model",
    )
    p_sup_predict.add_argument("--model-path", required=True)
    p_sup_predict.add_argument("--html", required=True, help="Path to candidate HTML")

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    detector = DOMOnlyRobustDetector()

    if args.cmd == "compare":
        ref_html = _read_text(args.reference_html)
        cand_html = _read_text(args.candidate_html)
        out = detector.compare_html(
            reference_html=ref_html,
            candidate_html=cand_html,
            reference_domain=args.reference_domain,
            candidate_domain=args.candidate_domain,
        )
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "detect":
        cand_html = _read_text(args.candidate_html)
        refs = _load_reference_map(args.references)
        out = detector.detect_against_references(
            candidate_html=cand_html,
            reference_html_map=refs,
            candidate_domain=args.candidate_domain,
        )
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "profile":
        html = _read_text(args.html)
        profile = detector.build_profile(html, page_domain=args.domain)
        print(json.dumps(profile.to_json_dict(), indent=2))
        return

    if args.cmd == "unsup-extract":
        clf = HTMLUnsupervisedDOMClassifier(hashed_dim=args.hashed_dim)
        files = clf.stream_split_to_feature_store(
            split_name=args.split,
            feature_dir=args.feature_dir,
            batch_size=args.batch_size,
            hf_token=args.hf_token,
            resume=args.resume,
        )
        print(json.dumps({"saved_chunks": files, "count": len(files)}, indent=2))
        return

    if args.cmd == "unsup-train":
        clf = HTMLUnsupervisedDOMClassifier(
            hashed_dim=args.hashed_dim,
            contamination=args.contamination,
        )
        info = clf.fit_from_feature_store_streaming(args.train_chunks)
        clf.save(args.model_path)
        info["model_path"] = args.model_path
        print(json.dumps(info, indent=2))
        return

    if args.cmd == "unsup-eval":
        clf = HTMLUnsupervisedDOMClassifier()
        clf.load(args.model_path)
        metrics = clf.evaluate_from_feature_store(args.test_chunks)
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "unsup-predict":
        clf = HTMLUnsupervisedDOMClassifier()
        clf.load(args.model_path)
        html = _read_text(args.html)
        pred = clf.predict_html(html)
        print(json.dumps(pred, indent=2))
        return

    if args.cmd == "sup-train":
        clf = HTMLSupervisedDOMClassifier(
            hashed_dim=args.hashed_dim,
            n_estimators=args.n_estimators,
        )
        info = clf.fit_from_feature_store_streaming(args.train_chunks)
        clf.save(args.model_path)
        info["model_path"] = args.model_path
        print(json.dumps(info, indent=2))
        return

    if args.cmd == "sup-eval":
        clf = HTMLSupervisedDOMClassifier()
        clf.load(args.model_path)
        metrics = clf.evaluate_from_feature_store(args.test_chunks)
        print(json.dumps(metrics, indent=2))
        return

    if args.cmd == "sup-predict":
        clf = HTMLSupervisedDOMClassifier()
        clf.load(args.model_path)
        html = _read_text(args.html)
        pred = clf.predict_html(html)
        print(json.dumps(pred, indent=2))
        return


if __name__ == "__main__":
    main()
