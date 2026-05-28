import argparse
import csv
import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class QueryEvaluation:

	query: str
	ground_truth: str
	predictions: List[str]
	rank: int
	average_precision: float
	f_measure: float

def extract_ground_truth(query_filename: str) -> str:
	"""Extract the expected DB filename from a query snippet filename.

	Example:
		classical.00000-snippet-10-0.wav -> classical.00000.wav

	Args:
		query_filename (str): Query audio filename, usually with a ``-snippet`` suffix.

	Returns:
		str: Ground-truth database filename expected for the query.
	"""
	if "-snippet" in query_filename:
		return query_filename.split("-snippet", 1)[0] + ".wav"
	return query_filename


def parse_output_file(output_path: str, top_k: int = 3) -> List[Tuple[str, str, List[str]]]:
	"""Parse output.txt style rows into ``(query, ground_truth, predictions)`` tuples.

	Args:
		output_path (str): Path to the tab-separated output file.
		top_k (int): Number of prediction columns to read from each row.

	Returns:
		list[tuple[str, str, list[str]]]: Parsed rows containing query filename,
			derived ground truth filename, and the top-k predictions.
	"""
	rows: List[Tuple[str, str, List[str]]] = []
	with open(output_path, "r", encoding="utf-8") as handle:
		for line_num, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue

			columns = line.split("\t")
			if len(columns) < 2:
				raise ValueError(
					f"Malformed row on line {line_num}: expected at least 2 columns, got {len(columns)}"
				)

			query = columns[0]
			predictions = columns[1 : 1 + top_k]
			ground_truth = extract_ground_truth(query)
			rows.append((query, ground_truth, predictions))

	return rows


def find_rank_of_correct_match(ground_truth: str, predictions: Sequence[str]) -> int:
	"""Find the 1-based rank where the ground truth appears in predictions.

	Args:
		ground_truth (str): Expected matching database filename.
		predictions (Sequence[str]): Ranked prediction list.

	Returns:
		int: Rank in ``[1..len(predictions)]`` if found, otherwise ``-1``.
	"""
	for index, prediction in enumerate(predictions, start=1):
		if prediction == ground_truth:
			return index
	return -1


def compute_average_precision(ground_truth: str, predictions: Sequence[str]) -> float:
	"""Compute Average Precision (AP) for a single query using standard IR metrics.

	Average Precision is the average of precision scores at each rank where
	a relevant document is retrieved. For top-k retrieval with a single relevant
	document, AP = (sum of precisions at relevant ranks) / 1.

	Args:

		ground_truth (str): Expected matching database filename.
		predictions (Sequence[str]): Ranked prediction list.
	Returns:
		float: Average Precision in ``[0, 1]``, or 0.0 if not found.
	"""
	rank = find_rank_of_correct_match(ground_truth, predictions)
	if rank == -1:
		return 0.0

	# For single relevant document: precision at rank = 1 / rank
	precision_at_rank = 1.0 / rank
	return precision_at_rank


def compute_f_measure(average_precision: float, rank: int) -> float:
	"""Compute F-measure using AP as precision and binary recall from rank.

	Args:
		average_precision (float): Average Precision value for the query.
		rank (int): Rank of the correct prediction, or ``-1`` if not found.

	Returns:
		float: F-measure (F1). Returns ``0.0`` when recall is zero.
	"""
	recall = 1.0 if rank != -1 else 0.0
	denom = average_precision + recall
	if denom == 0.0:
		return 0.0
	return (2.0 * average_precision * recall) / denom

def evaluate_query(query: str, ground_truth: str, predictions: Sequence[str]) -> QueryEvaluation:
	"""Evaluate one query and produce rank and Average Precision (AP).

	Args:
		query (str): Query filename.
		ground_truth (str): Expected database filename for the query.
		predictions (Sequence[str]): Ranked predicted filenames.

	Returns:
		QueryEvaluation: Structured per-query evaluation with AP.

	"""
	rank = find_rank_of_correct_match(ground_truth, predictions)
	ap = compute_average_precision(ground_truth, predictions)
	f_measure = compute_f_measure(ap, rank)
	return QueryEvaluation(
		query=query,
		ground_truth=ground_truth,
		predictions=list(predictions),
		rank=rank,
		average_precision=ap,
		f_measure=f_measure,
	)

def _safe_mean(values: Sequence[float]) -> float:
	"""Compute mean, returning ``0.0`` for an empty sequence.

	Args:
		values (Sequence[float]): Numeric values.

	Returns:
		float: Arithmetic mean or ``0.0`` if no values are provided.
	"""
	return statistics.fmean(values) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
	"""Compute population standard deviation safely.

	Args:
		values (Sequence[float]): Numeric values.

	Returns:
		float: Population standard deviation, or ``0.0`` if fewer than two values.
	"""
	if len(values) < 2:
		return 0.0
	return statistics.pstdev(values)


def aggregate_results(evaluations: Sequence[QueryEvaluation]) -> Dict[str, object]:
	"""Aggregate per-query results into summary metrics including Mean Average Precision (MAP).

	Args:
		evaluations (Sequence[QueryEvaluation]): Per-query evaluation objects.

	Returns:

		dict[str, object]: Summary including Mean Average Precision (MAP),
			standard deviation, coverage, and rank distribution.
	"""
	average_precisions = [item.average_precision for item in evaluations]
	f_measures = [item.f_measure for item in evaluations]
	rank_distribution = {1: 0, 2: 0, 3: 0, -1: 0}
	for item in evaluations:
		if item.rank in rank_distribution:
			rank_distribution[item.rank] += 1
		else:
			rank_distribution[item.rank] = rank_distribution.get(item.rank, 0) + 1

	total = len(evaluations)
	covered = total - rank_distribution.get(-1, 0)
	coverage = (covered / total) if total else 0.0

	return {
		"num_queries": total,
		"mean_average_precision": _safe_mean(average_precisions),
		"std_average_precision": _safe_std(average_precisions),
		"mean_f_measure": _safe_mean(f_measures),
		"std_f_measure": _safe_std(f_measures),
		"coverage": coverage,
		"rank_distribution": rank_distribution,

	}

def evaluate_output_file(output_path: str, top_k: int = 3) -> Dict[str, object]:
	"""Evaluate all queries in an output file and compute aggregate statistics.

	Args:
		output_path (str): Path to the system output file.
		top_k (int): Number of predictions per row to evaluate.

	Returns:
		dict[str, object]: Dictionary with ``per_query`` and ``aggregate`` keys.
	"""
	parsed_rows = parse_output_file(output_path, top_k=top_k)
	per_query = [evaluate_query(query, gt, preds) for query, gt, preds in parsed_rows]
	aggregate = aggregate_results(per_query)
	return {
		"per_query": per_query,
		"aggregate": aggregate,
	}


def write_per_query_csv(evaluations: Sequence[QueryEvaluation], csv_path: str) -> None:
	"""Write per-query evaluation metrics (including AP) to a CSV file.

	Args:
		evaluations (Sequence[QueryEvaluation]): Per-query evaluation records.
		csv_path (str): Destination CSV path.

	Returns:
		None: Writes CSV output to disk.
	"""
	with open(csv_path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(
			[
				"query",
				"ground_truth",
				"pred_1",
				"pred_2",
				"pred_3",
				"rank",
				"average_precision",
				"f_measure",
			]
		)

		for item in evaluations:
			preds = item.predictions + [""] * max(0, 3 - len(item.predictions))
			writer.writerow(
				[
					item.query,
					item.ground_truth,
					preds[0],
					preds[1],
					preds[2],
					item.rank,
					f"{item.average_precision:.6f}",
					f"{item.f_measure:.6f}",
				]
			)

def print_report(results: Dict[str, object]) -> None:
	"""Print a human-readable summary report of evaluation metrics including MAP.

	Args:
		results (dict[str, object]): Result dictionary from ``evaluate_output_file``.

	Returns:
		None: Writes formatted output to stdout.
	"""

	agg = results["aggregate"]
	print("Audio Identification Evaluation")
	print("=" * 40)
	print(f"Queries evaluated: {agg['num_queries']}")
	print(f"Mean Average Precision (MAP): {agg['mean_average_precision']:.4f} +/- {agg['std_average_precision']:.4f}")
	print(f"Mean F-measure: {agg['mean_f_measure']:.4f} +/- {agg['std_f_measure']:.4f}")
	print(f"Top-k coverage: {agg['coverage']:.4f}")
	print("Rank distribution:")
	rank_dist = agg["rank_distribution"]
	print(f"  rank 1: {rank_dist.get(1, 0)}")
	print(f"  rank 2: {rank_dist.get(2, 0)}")
	print(f"  rank 3: {rank_dist.get(3, 0)}")
	print(f"  miss:   {rank_dist.get(-1, 0)}")

def main() -> None:
	"""Run the command-line interface for evaluating output files.

	Args:
		None: Uses command-line arguments from ``sys.argv``.

	Returns:
		None: Prints report and optionally writes CSV.
	"""
	parser = argparse.ArgumentParser(description="Evaluate audio identification output metrics")
	parser.add_argument("--output", required=True, help="Path to output.txt")
	parser.add_argument("--top-k", type=int, default=3, help="Number of predictions per row to evaluate")
	parser.add_argument(
		"--csv",
		required=False,
		help="Optional path to save per-query metrics CSV",
	)
	args = parser.parse_args()

	if args.top_k < 1:
		raise ValueError("--top-k must be >= 1")

	results = evaluate_output_file(args.output, top_k=args.top_k)
	print_report(results)

	if args.csv:
		write_per_query_csv(results["per_query"], args.csv)
		print(f"Saved per-query metrics to: {args.csv}")


if __name__ == "__main__":
	main()
