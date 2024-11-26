# Load the provided file and process it to calculate evaluation metrics
file_path = './log.txt'
file_path = './test_results.txt'

# Read the file content
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Initialize counters
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# Process each line
for line in lines:
    if not line.strip():
        continue
    file_path, prediction = line.split(':')
    file_path = file_path.strip()
    prediction = prediction.strip()

    # Determine the ground truth
    if '/smoking_' in file_path or 'a00' in file_path:
        ground_truth = '흡연'
    elif 'notsmoking' in file_path or 'gg' in file_path or 'image' in file_path:
        ground_truth = '비흡연'
    else:
        continue

    # Compare prediction with ground truth
    if ground_truth == '흡연' and prediction == '흡연':
        true_positive += 1
    elif ground_truth == '비흡연' and prediction == '비흡연':
        true_negative += 1
    elif ground_truth == '비흡연' and prediction == '흡연':
        false_positive += 1
    elif ground_truth == '흡연' and prediction == '비흡연':
        print(ground_truth,prediction)
        false_negative += 1

# Calculate metrics
total_predictions = true_positive + true_negative + false_positive + false_negative
accuracy = (true_positive + true_negative) / total_predictions
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = (
    2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
)
print(
{
    "true_positive": true_positive,
    "true_negative": true_negative,
    "false_positive": false_positive,
    "false_negative": false_negative,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
}
)