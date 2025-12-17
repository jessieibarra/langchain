"""
Classification accuracy evaluator - checks if classify_intent predicted the correct intent.
"""


def classification_accuracy(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """
    Evaluate if the classification intent was correct.

    Args:
        inputs: Input data
        outputs: Graph outputs (includes classification with intent)
        reference_outputs: Expected outputs (includes expected_intent)

    Returns:
        dict with accuracy score (1.0 if correct, 0.0 if wrong)
    """
    # Get the classification from graph output
    classification = outputs.get("classification", {})
    predicted_intent = classification.get("intent")

    if predicted_intent is None:
        return {
            "key": "classification_accuracy",
            "score": 0.0,
            "comment": "No classification found in outputs"
        }

    # Check if we have reference data
    if reference_outputs:
        expected_intent = reference_outputs.get("expected_intent")

        if expected_intent is None:
            return {
                "key": "classification_accuracy",
                "score": 0.5,
                "comment": "No expected_intent in reference outputs"
            }

        # Check if prediction matches expected
        is_correct = predicted_intent == expected_intent

        return {
            "key": "classification_accuracy",
            "score": 1.0 if is_correct else 0.0,
            "comment": f"Predicted: {predicted_intent}, Expected: {expected_intent}" + (" ✓" if is_correct else " ✗")
        }
    else:
        # No reference data - can't evaluate accuracy
        return {
            "key": "classification_accuracy",
            "score": 0.5,
            "comment": f"Predicted: {predicted_intent} (no reference to compare)"
        }

