"""
Main evaluation runner - compose and run evaluation suites.

Usage:
    # Run all evaluators
    python -m agent.run_evaluation

    # Run specific evaluators
    python -m agent.run_evaluation --evaluators playlist_quality

    # Run with custom dataset
    python -m agent.run_evaluation --dataset my-dataset
"""

import argparse
from langsmith import evaluate
from agent.graph import graph
from agent.evaluators import playlist_quality, conversation_tone, classification_accuracy


def target(inputs: dict) -> dict:
    """Run the graph with dataset inputs."""
    return graph.invoke(inputs)


# Define evaluation suites
EVALUATION_SUITES = {
    "all": [playlist_quality, conversation_tone, classification_accuracy],
    "playlist": [playlist_quality],
    "tone": [conversation_tone],
    "classification": [classification_accuracy],
    "combined": [playlist_quality, conversation_tone, classification_accuracy],  # alias for "all"
}


def main():
    parser = argparse.ArgumentParser(description="Run DJ Agent evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="playlist-golden-dataset",
        help="Dataset name or ID to evaluate on",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=list(EVALUATION_SUITES.keys()),
        default="all",
        help="Evaluation suite to run",
    )
    parser.add_argument(
        "--evaluators",
        type=str,
        nargs="+",
        choices=["playlist_quality", "conversation_tone", "classification_accuracy"],
        help="Specific evaluators to run (overrides --suite)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Experiment prefix (default: auto-generated)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per example",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Maximum concurrent evaluations",
    )

    args = parser.parse_args()

    # Determine which evaluators to use
    if args.evaluators:
        evaluator_map = {
            "playlist_quality": playlist_quality,
            "conversation_tone": conversation_tone,
            "classification_accuracy": classification_accuracy,
        }
        evaluators = [evaluator_map[e] for e in args.evaluators]
        suite_name = "-".join(args.evaluators)
    else:
        evaluators = EVALUATION_SUITES[args.suite]
        suite_name = args.suite

    # Generate experiment prefix if not provided
    if args.prefix is None:
        prefix = f"dj-{suite_name}"
        if args.repetitions > 1:
            prefix += f"-rep{args.repetitions}"
    else:
        prefix = args.prefix

    print(f"Running evaluation:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Evaluators: {[e.__name__ for e in evaluators]}")
    print(f"  Repetitions: {args.repetitions}")
    print(f"  Experiment prefix: {prefix}")
    print()

    # Run evaluation
    results = evaluate(
        target,
        data=args.dataset,
        evaluators=evaluators,
        experiment_prefix=prefix,
        num_repetitions=args.repetitions,
        max_concurrency=args.max_concurrency,
    )

    print("\nEvaluation complete!")
    print(results)


if __name__ == "__main__":
    main()

