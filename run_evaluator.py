import argparse
import numpy as np
import pandas as pd
import json
import logging
from src.models import ModelType, EvaluationConfig
from src.openai_evaluator import OpenAIEvaluator
from src.deepinfra_evaluator import DeepInfraEvaluator
from src.claude_evaluator import ClaudeEvaluator
from src.eval_metrics import calculate_metrics

logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='L3-8b')
    parser.add_argument('--input_path', type=str, default='data/EUDisinfoTest.csv')
    parser.add_argument('--output_path', type=str, default='evaluation_results_L3_8b.csv')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--max_trials', type=int, default=3)
    args = parser.parse_args()
    return EvaluationConfig(args.model, args.input_path, args.output_path, args.api_key, args.max_trials)


def initialize_evaluator(model_name, api_key):
    if model_name in ['L3-8b', 'L3-70b', 'Mixtral']:
        return DeepInfraEvaluator(ModelType[model_name], api_key=api_key)
    elif model_name in ['GPT3.5', 'GPT4o']:
        return OpenAIEvaluator(ModelType[model_name], api_key=api_key)
    elif model_name in ['Sonnet', 'Haiku', 'Opus']:
        return ClaudeEvaluator(ModelType[model_name], api_key=api_key)
    else:
        raise ValueError(f"Model {model_name} not supported.")


def evaluate_data(evaluator, data_df, max_trials=3):
    results = {'base': [], 'logos': [], 'ethos': [], 'pathos': []}
    for _, row in data_df.iterrows():
        trial_count = 0
        while trial_count < max_trials:
            try:
                prediction, ground_truth = evaluate_row(evaluator, row)
                category = determine_category(row)
                results[category].append((prediction, ground_truth))
                break
            except Exception as e:
                logging.error(
                    f"Attempt {trial_count + 1}: Error during evaluation for narrative: {row['narrative']} - {e}")
                trial_count += 1
    return results


def evaluate_row(evaluator, row):
    result = evaluator.evaluate(row['narrative'])
    prediction = 0 if result['status'] == 'disinformative' else 1
    ground_truth = 1 if row['credible'] else 0
    return prediction, ground_truth


def determine_category(row):
    if row['logos']:
        return 'logos'
    elif row['ethos']:
        return 'ethos'
    elif row['pathos']:
        return 'pathos'
    return 'base'


def calculate_aggregate_metrics(results):
    aggregate_metrics = {}
    agg_f1_score = []
    agg_tnr = []
    agg_tpr = []
    for key, pairs in results.items():

        if pairs:
            predictions, ground_truths = zip(*pairs)
            metrics = calculate_metrics(ground_truths, predictions)
            aggregate_metrics.update({f'F1-Score-{key.capitalize()}': metrics['F1-Score'],
                                      f'TNR-{key.capitalize()}': metrics['TNR'],
                                      f'TPR-{key.capitalize()}': metrics['TPR']})
            agg_f1_score.append(metrics['F1-Score'])
            agg_tnr.append(metrics['TNR'])
            agg_tpr.append(metrics['TPR'])
    aggregate_metrics.update({'Agg-F1-Score': np.mean(agg_f1_score),
                              'Agg-TNR': np.mean(agg_tnr),
                              'Agg-TPR': np.mean(agg_tpr)})
    return aggregate_metrics


def main():
    config = parse_arguments()
    data_df = pd.read_csv(config.input_path)
    evaluator = initialize_evaluator(config.model, config.api_key)
    results = evaluate_data(evaluator, data_df, max_trials=config.max_trials)
    final_metrics = calculate_aggregate_metrics(results)
    logging.info(f"Evaluation Metrics: {final_metrics}")
    with open(f"{config.output_path}", 'w') as f:
        json.dump(final_metrics, f)


if __name__ == "__main__":
    main()
