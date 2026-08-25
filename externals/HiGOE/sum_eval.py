import argparse

from tqdm import tqdm
from bert_score import score
from rouge_score import rouge_scorer

from utils import *

def bert_score_eval(generate_response, ground_truth, device, batch_size=8):
    P, R, F = score(generate_response, ground_truth, model_type="microsoft/deberta-xlarge-mnli", device=device,
                    batch_size=batch_size)
    P = [float(i) for i in P.numpy()]
    R = [float(i) for i in R.numpy()]
    F = [float(i) for i in F.numpy()]

    return P, R, F


def rouge_eval(generate_response, ground_truth, type='rougeL'):
    if not isinstance(ground_truth, str):
        num_ref = len(ground_truth)
        generate_response_expand = [generate_response] * num_ref
        ground_truth_expand = ground_truth
        Ps = []
        Rs = []
        Fs = []
        for i, j in zip(generate_response_expand, ground_truth_expand):
            scorer = rouge_scorer.RougeScorer([type], use_stemmer=True)
            scores = scorer.score(prediction=i, target=j)
            Ps.append(scores[type].precision)
            Rs.append(scores[type].recall)
            Fs.append(scores[type].fmeasure)
        P = max(Ps)
        R = max(Rs)
        F = max(Fs)

        return float(P), float(R), float(F)
    else:
        scorer = rouge_scorer.RougeScorer([type], use_stemmer=True)
        scores = scorer.score(prediction=generate_response, target=ground_truth)
        P = scores[type].precision
        R = scores[type].recall
        F = scores[type].fmeasure

        return float(P), float(R), float(F)


def response_eval(generate_responses, ground_truthes):
    metric_list = []
    for ind, (generate_response, ground_truth) in enumerate(tqdm(zip(generate_responses, ground_truthes))):
        metrics = dict()
        _, _, rouge_L_F = rouge_eval(generate_response, ground_truth, type='rougeL')
        _, _, rouge_1_F = rouge_eval(generate_response, ground_truth, type='rouge1')
        _, _, rouge_2_F = rouge_eval(generate_response, ground_truth, type='rouge2')
        metrics["ROUGE-L"] = {"F": rouge_L_F}
        metrics["ROUGE-1"] = {"F": rouge_1_F}
        metrics["ROUGE-2"] = {"F": rouge_2_F}
        metric_list.append(metrics)

    all_metrics = dict()
    for key in metric_list[0].keys():
        all_metrics[key] = {kk: float(np.mean([vv[key][kk] for vv in metric_list])) for kk in
                            metric_list[0][key].keys()}

    print("\n")
    print(text_wrap("=" * 50 + "Final Evaluation" + "=" * 50))
    print_metrics(all_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    opt = parser.parse_args()
    FILE_NAME = opt.file_name
    SEED = opt.seed
    set_seed(int(SEED))
    DEVICE = get_device(int(opt.cuda))

    with open(FILE_NAME, 'r') as file:
        doc_data = json.load(file)

    print("{} #Test Data: {}".format(show_time(), len(doc_data)))

    generate_responses, ground_truthes = [], []
    for q, v in doc_data.items():
        generate_responses.append(v["response"])
        ground_truthes.append(v["gt"])

    response_eval(generate_responses=generate_responses, ground_truthes=ground_truthes)
