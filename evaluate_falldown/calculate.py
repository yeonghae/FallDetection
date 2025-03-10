"""
추정된 포즈 결과로부터 전체(낮, 밤), 영상별 정확도를 계산
"""

import os
import json
import random

def load_label_list(label_dir):
    label_list = []
    file_list = os.listdir(label_dir)
    for f in file_list:
        if f.endswith('.json'):
            label_list.append(f)
    return label_list

def load_json(label_dir, result_dir, method, filename):
    label_filepath = os.path.join(label_dir, filename)
    result_filepath = os.path.join(result_dir, method, filename)

    with open(label_filepath, 'r') as f:
        label_data = json.load(f)
    
    with open(result_filepath, 'r') as f:
        result_data = json.load(f)
    
    return label_data, result_data

def calculate_accuracy(label_data, result_data):
    total = max(len(label_data), len(result_data))
    search = min(len(label_data), len(result_data))
    correct = 0
    weight = total - search
    
    for i in range(search):
        if label_data[i] == result_data[i]:
            correct += 1
    
    accuracy = (correct + weight) / total
    return accuracy

def calculate_summary(day_accuracy, night_accuracy):
    day_accuracy = [d['accuracy'] for d in day_accuracy]
    night_accuracy = [n['accuracy'] for n in night_accuracy]
    summary = {
        'day': {
            'total': len(day_accuracy),
            'accuracy': sum(day_accuracy) / len(day_accuracy),
        },
        'night': {
            'total': len(night_accuracy),
            'accuracy': sum(night_accuracy) / len(night_accuracy),
        },
    }
    return summary

def save(result):
    with open('calculate_result.json', 'w') as f:
        json.dump(result, f, indent=4)

def main():
    label_dir = 'dataset/labels'
    result_dir = 'result'

    calculate_result = {
        'summary': None,
        'day': None,
        'night': None,
    }
    day_accuracy = []
    night_accuracy = []

    methods = ['day', 'night']

    label_list = load_label_list(label_dir)

    for method in methods:
        for filename in label_list:
            label_data, result_data = load_json(label_dir, result_dir, method, filename)
            accuracy = calculate_accuracy(label_data, result_data)
            calculate_data = {'video': f"dataset/{method}/{filename.split('.')[0]}.mp4", 'accuracy': accuracy}
            print(calculate_data)
            if method == 'day':
                day_accuracy.append(calculate_data)
            else:
                night_accuracy.append(calculate_data)

    calculate_result['summary'] = calculate_summary(day_accuracy, night_accuracy)
    calculate_result['day'] = day_accuracy
    calculate_result['night'] = night_accuracy
    save(calculate_result)
    print(calculate_result['summary'])
    print("Done")

if __name__ == "__main__":
    main()