from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from datasets import load_dataset
from playwright.sync_api import BrowserContext, Page, sync_playwright


def _normalize_label(raw) -> int:
	"""Normalize label values from dataset to 0 (benign) or 1 (phishing)."""
	try:
		if isinstance(raw, str):
			return 1 if raw.lower() in {"phish", "phishing", "1"} else 0
		return 1 if int(raw) == 1 else 0
	except Exception:
		return 0


def fetch_one_per_class(split: str, hf_token: Optional[str]) -> Dict[int, Dict[str, str]]:
	"""Stream dataset and return one benign and one phishing sample."""
	found: Dict[int, Dict[str, str]] = {}
	ds = load_dataset("phreshphish/phreshphish", split=split, streaming=True, token=hf_token)

	for sample in ds:
		label = _normalize_label(sample.get("label", 0))
		if label in found:
			continue

		found[label] = {
			"html": sample.get("html", "") or "",
			"label_raw": str(sample.get("label", "")),
			"url": str(sample.get("url", "")),
		}

		if 0 in found and 1 in found:
			break

	missing = [str(lbl) for lbl in (0, 1) if lbl not in found]
	if missing:
		raise RuntimeError(f"Could not find required classes in split '{split}': {', '.join(missing)}")

	return found


def _iter_dataset_samples(splits: Iterable[str], hf_token: Optional[str]):
	"""Yield samples from one or more splits in sequence."""
	for split in splits:
		ds = load_dataset("phreshphish/phreshphish", split=split, streaming=True, token=hf_token)
		for sample in ds:
			yield split, sample


def _normalize_url(raw_url: str) -> str:
	url = (raw_url or "").strip()
	if not url:
		return ""
	if not (url.startswith("http://") or url.startswith("https://")):
		url = "http://" + url
	return url


def _is_reachable_and_capture(
	page: Page,
	url: str,
	screenshot_path: Path,
	navigation_timeout_ms: int,
) -> Tuple[bool, Optional[int], str]:
	"""Try opening URL and capture screenshot only on success."""
	try:
		page.set_default_navigation_timeout(navigation_timeout_ms)
		response = page.goto(url, wait_until="domcontentloaded")
		status = response.status if response is not None else None

		# Treat unknown status as unreachable for conservative filtering.
		if status is None:
			return False, None, "no_response"

		if status >= 400:
			return False, status, f"http_{status}"

		# Allow short time for dynamic DOM changes before screenshot.
		page.wait_for_timeout(1200)
		screenshot_path.parent.mkdir(parents=True, exist_ok=True)
		page.screenshot(path=str(screenshot_path), full_page=True)
		return True, status, "ok"
	except Exception as exc:
		msg = str(exc).replace("\n", " ")
		return False, None, msg[:240]


def _build_filename(label_name: str, idx: int, url: str) -> str:
	safe = url.replace("https://", "").replace("http://", "")
	safe = safe.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_")
	if len(safe) > 140:
		safe = safe[:140]
	return f"{label_name}_{idx:06d}_{safe}.png"


def collect_live_samples(
	splits: List[str],
	hf_token: Optional[str],
	out_dir: Path,
	target_label: int,
	target_count: int,
	max_scan: int,
	navigation_timeout_ms: int,
	viewport_w: int,
	viewport_h: int,
) -> Dict[str, int]:
	"""Collect live reachable URLs by label and capture screenshots."""
	out_dir.mkdir(parents=True, exist_ok=True)
	screenshot_dir = out_dir / ("phishing" if target_label == 1 else "benign")
	results_csv = out_dir / f"live_results_{'phishing' if target_label == 1 else 'benign'}.csv"
	results_jsonl = out_dir / f"live_results_{'phishing' if target_label == 1 else 'benign'}.jsonl"

	fieldnames = [
		"scan_index",
		"split",
		"label",
		"url",
		"reachable",
		"status",
		"reason",
		"screenshot_path",
	]

	seen_urls: Set[str] = set()
	scanned = 0
	label_seen = 0
	reachable_count = 0

	with results_csv.open("w", newline="", encoding="utf-8") as csv_f, results_jsonl.open("w", encoding="utf-8") as jsonl_f:
		writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
		writer.writeheader()

		with sync_playwright() as p:
			browser = p.chromium.launch(headless=True)
			context: BrowserContext = browser.new_context(viewport={"width": viewport_w, "height": viewport_h})
			page: Page = context.new_page()

			for split, sample in _iter_dataset_samples(splits, hf_token):
				scanned += 1
				if scanned > max_scan:
					break

				label = _normalize_label(sample.get("label", 0))
				if label != target_label:
					continue

				raw_url = str(sample.get("url", ""))
				url = _normalize_url(raw_url)
				if not url:
					continue

				if url in seen_urls:
					continue
				seen_urls.add(url)
				label_seen += 1

				filename = _build_filename("phishing" if target_label == 1 else "benign", label_seen, url)
				screenshot_path = screenshot_dir / filename

				reachable, status, reason = _is_reachable_and_capture(
					page=page,
					url=url,
					screenshot_path=screenshot_path,
					navigation_timeout_ms=navigation_timeout_ms,
				)

				if reachable:
					reachable_count += 1
				else:
					screenshot_path = Path("")

				row = {
					"scan_index": scanned,
					"split": split,
					"label": label,
					"url": url,
					"reachable": int(reachable),
					"status": "" if status is None else status,
					"reason": reason,
					"screenshot_path": str(screenshot_path),
				}
				writer.writerow(row)
				jsonl_f.write(json.dumps(row, ensure_ascii=True) + "\n")

				if label_seen % 100 == 0:
					print(
						(
							f"[collect-live] label_seen={label_seen} reachable={reachable_count} "
							f"ratio={reachable_count / max(label_seen, 1):.3f} scanned={scanned}"
						),
						flush=True,
					)

				if reachable_count >= target_count:
					break

			browser.close()

	return {
		"target_label": target_label,
		"target_count": target_count,
		"reachable_collected": reachable_count,
		"label_seen": label_seen,
		"scanned_total": scanned,
		"results_csv": str(results_csv),
		"results_jsonl": str(results_jsonl),
		"screenshot_dir": str(screenshot_dir),
	}


def render_html_to_png(html: str, output_png: Path, viewport_w: int, viewport_h: int, allow_network: bool) -> None:
	"""Render raw HTML into a screenshot using Playwright."""
	output_png.parent.mkdir(parents=True, exist_ok=True)

	with sync_playwright() as p:
		browser = p.chromium.launch(headless=True)
		context = browser.new_context(viewport={"width": viewport_w, "height": viewport_h})

		if not allow_network:
			# Keep rendering deterministic and safe by blocking external requests.
			context.route("**/*", lambda route, request: route.abort() if request.resource_type != "document" else route.continue_())

		page = context.new_page()
		page.set_content(html, wait_until="domcontentloaded")
		page.wait_for_timeout(1200)
		page.screenshot(path=str(output_png), full_page=True)

		browser.close()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Sample and collect screenshots from phreshphish dataset."
	)
	parser.add_argument("--mode", default="one-per-class", choices=["one-per-class", "collect-live"])
	parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split for one-per-class mode")
	parser.add_argument(
		"--splits",
		nargs="+",
		default=["train", "test"],
		choices=["train", "test"],
		help="Dataset splits for collect-live mode",
	)
	parser.add_argument("--out-dir", default="./sample_preview", help="Output directory for HTML and PNG files")
	parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token")
	parser.add_argument("--viewport-width", type=int, default=1366)
	parser.add_argument("--viewport-height", type=int, default=900)
	parser.add_argument("--target", default="phishing", choices=["phishing", "benign"], help="Target class for collect-live mode")
	parser.add_argument("--target-count", type=int, default=5000, help="How many reachable screenshots to collect")
	parser.add_argument("--max-scan", type=int, default=200000, help="Max streamed samples to inspect before stopping")
	parser.add_argument("--navigation-timeout-ms", type=int, default=12000, help="Per-URL timeout in milliseconds")
	parser.add_argument(
		"--allow-network",
		action="store_true",
		help="Used only in one-per-class mode. Allow external requests while rendering HTML content.",
	)
	args = parser.parse_args()

	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	if args.mode == "collect-live":
		target_label = 1 if args.target == "phishing" else 0
		summary = collect_live_samples(
			splits=args.splits,
			hf_token=args.hf_token,
			out_dir=out_dir,
			target_label=target_label,
			target_count=args.target_count,
			max_scan=args.max_scan,
			navigation_timeout_ms=args.navigation_timeout_ms,
			viewport_w=args.viewport_width,
			viewport_h=args.viewport_height,
		)
		print(json.dumps(summary, indent=2))
		return

	samples = fetch_one_per_class(split=args.split, hf_token=args.hf_token)

	for label in (0, 1):
		sample = samples[label]
		class_name = "benign" if label == 0 else "phishing"

		html_path = out_dir / f"sample_{class_name}.html"
		png_path = out_dir / f"sample_{class_name}.png"

		html_path.write_text(sample["html"], encoding="utf-8", errors="ignore")
		render_html_to_png(
			html=sample["html"],
			output_png=png_path,
			viewport_w=args.viewport_width,
			viewport_h=args.viewport_height,
			allow_network=args.allow_network,
		)

		print(f"[{class_name}] label_raw={sample['label_raw']} url={sample['url']}")
		print(f"[{class_name}] html={html_path}")
		print(f"[{class_name}] screenshot={png_path}")


if __name__ == "__main__":
	main()
