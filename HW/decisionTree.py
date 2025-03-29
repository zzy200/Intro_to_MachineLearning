import sys
import math
from collections import Counter

def load_data(input_file):
    with open(input_file, 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    headers = lines[0]
    #print(f"Headers: {headers}")
    features = [line[:-1] for line in lines[1:]] #all features values except the last one(labels)
    labels = [line[-1] for line in lines[1:]]
    return headers, features, labels

def calculate_entropy(labels):
    total = len(labels)
    if total == 0.0:
        return 0.0
    counts = Counter(labels)
    entropy = 0.0
    for label in counts:
        prob = counts[label] / total
        entropy -= prob * math.log2(prob)
    return entropy

def calculate_mutual_information(features, labels, attribute_index):
    #attribute index is the index of the current feature being considered

    total = len(labels)
    parent_entropy = calculate_entropy(labels)
    #print(f"{attribute_index}")
    #for feature in features:
        #print(f"{len(feature)}") 
    split_value = set(feature[attribute_index] for feature in features)
    weighted_child_entropy = 0.0

    for value in split_value:
        subset_labels = [label for feature, label in zip(features, labels) if feature[attribute_index] == value]
        weighted_child_entropy += (len(subset_labels) / total) * calculate_entropy(subset_labels)
    
    mutual_information = parent_entropy - weighted_child_entropy
    return mutual_information

class TreeNode:
    def __init__(self, attribute=None, value=None, left=None, right=None, label_counts=None):
        self.attribute = attribute  # 分裂属性（如 "Anti_satellite_test_ban"）
        self.value = value          # 分裂值（如 "y" 或 "n"）
        self.left = left            # 左子树（对应 value=True）
        self.right = right          # 右子树（对应 value=False）
        self.label_counts = label_counts  # 当前节点的标签统计（如 {"democrat": 10}）

def build_tree(features, labels, headers, depth, max_depth, attr=None, value=None):

    label_counts = Counter(labels)
    if depth >= max_depth or len(set(labels)) == 1:
        return TreeNode(attribute=attr, value=value, left=None, right=None, label_counts=label_counts)
    
    valid_attrs = [
        i for i in range(len(headers) - 1) 
        if len(set(feat[i] for feat in features)) > 1  # 只保留取值>1的属性
    ]

    # 如果没有有效属性可分裂，提前终止
    if not valid_attrs:
        return TreeNode(attribute=attr, value=value, label_counts=label_counts)
    
    # 从有效属性中选择最佳分裂点
    best_attr_index = max(
        valid_attrs,
        key=lambda i: calculate_mutual_information(features, labels, i)
    )
    best_attr = headers[best_attr_index]
    best_attr_values_set = set(feature[best_attr_index] for feature in features)
    best_attr_values_list = list(best_attr_values_set)


    #print(f"Best attribute 0: {best_attr_values_list}")
    left_labels, left_features = [], []
    right_labels, right_features = [], []
    for feature, label in zip(features, labels):
        if feature[best_attr_index] == best_attr_values_list[0]:
            left_labels.append(label)
            left_features.append(feature)
        else:
            right_labels.append(label)
            right_features.append(feature)
    
    left = build_tree(left_features, left_labels, headers, depth + 1, max_depth, best_attr, best_attr_values_list[0])
    right = build_tree(right_features, right_labels, headers, depth + 1, max_depth, best_attr, best_attr_values_list[1])

    return TreeNode(attribute=attr, value=value, left=left, right=right, label_counts=label_counts)

def predict(tree, sample, headers):
    if tree.left is None and tree.right is None:
        #print(f"Reached Leaf node: {tree.label_counts}")
        return max(tree.label_counts, key=tree.label_counts.get)  # 返回出现次数最多的标签
    
    attr_index = headers.index(tree.left.attribute)
    if sample[attr_index] == tree.left.value:
        #print("going left")
        return predict(tree.left, sample, headers)
    else:
        #print("going right")
        return predict(tree.right, sample, headers)

def write_predictions(tree, features, headers, output_file):
    with open(output_file, 'w') as f:
        for feature in features:
            pred = predict(tree, feature, headers)
            f.write(f"{pred}\n")
            

def print_tree(node, depth=0, labels_set=None):

    if labels_set is None:
        labels_set = set(node.label_counts.keys())
    sorted_labels = sorted(labels_set)  # 按字母顺序排序
    
    parts = []
    for label in sorted_labels:
        count = node.label_counts.get(label, 0)  # 如果标签不存在，默认为0
  
        parts.append(f"{count} {label}")
    formated_counts = "[" + "/".join(parts) + "]"

    prefix = "| " * depth
    if depth == 0:
        print(f"{formated_counts}")
        if node.left:
            print_tree(node.left, depth + 1, labels_set)
        if node.right:
            print_tree(node.right, depth + 1, labels_set)
    else:
        print(f"{prefix}{node.attribute} = {node.value}: {formated_counts}")
        if node.left:
            print_tree(node.left, depth + 1, labels_set)
        if node.right:
            print_tree(node.right, depth + 1, labels_set)

def calculate_error(true_labels, predicted_labels):
    return sum(1 for true, pred in zip(true_labels, predicted_labels) if true != pred) / len(true_labels)

def main(train_input, test_input, max_depth, train_output, test_output, metrics_output):
    #load data
    headers, train_features, train_labels = load_data(train_input)
    _, test_features, test_labels = load_data(test_input)
    labels_set = set(train_labels)

    #train the decision tree
    tree = build_tree(train_features, train_labels, headers, 0, max_depth)

    #write the predictions to the output files
    write_predictions(tree, train_features, headers, train_output)
    write_predictions(tree, test_features, headers, test_output)

    #calculate the metrics
    train_predicted_labels = [predict(tree, feature, headers) for feature in train_features]
    test_predicted_labels = [predict(tree, feature, headers) for feature in test_features]
    with open(metrics_output, 'w') as f:
        f.write(f"error(train): {calculate_error(train_labels, train_predicted_labels)}\n")
        f.write(f"error(test): {calculate_error(test_labels, test_predicted_labels)}\n")
    
    #print the decision tree
    print_tree(tree, 0, labels_set)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 decisionTree.py <train_input> <test_input> <max_depth> <train_output> <test_output> <metrics_output>")
        sys.exit(1)
    
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_output = sys.argv[4]
    test_output = sys.argv[5]
    metrics_output = sys.argv[6]

    main(train_input, test_input, max_depth, train_output, test_output, metrics_output)
    