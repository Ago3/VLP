import json
import itertools


def eval_file(answers, dataset):
    print(answers, dataset)
    tp, tn, fp, fn = 0, 0, 0, 0
    with open(answers, 'r') as f, open(dataset, 'r') as d:
        ans = json.loads(f.read())
        answers = dict()
        for a in ans:
            answers[a['question_id']] = a['answer']
        for i, line in enumerate(d.readlines()):
            if i == 0 and i not in answers:
                continue
            gold = 1 if line.split('\t')[3] == 'ITSELF' else 0
            if i not in answers:
                print(i)
                continue
            if gold and answers[i]:
                tp += 1
            elif gold:
                fn += 1
            elif answers[i]:
                fp += 1
                # print(line)
                # print(answers[i])
                # input()
            else:
                tn += 1
    precision = tp / max([(tp + fp), 0.01])
    recall = tp / max([(tp + fn), 0.01])
    f1 = 2 * precision * recall / max([(precision + recall), 0.01])
    print("\nPrecision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print(tp, fp, tn, fn)
    return precision, recall, f1


def score(epochs):
    with open('cc_log.tsv', 'w+') as log:
        for epoch in epochs:
            print("----\nEPOCH: ", epoch)
            log_line = '{}\t'.format(epoch)
            for split in ['val', 'test', 'test_hard']:
                answers = 'cc_{}_{}.json'.format(split, epoch)
                dataset = '../Image2Synset/DATA/15_{}.tsv.tmp'.format(split)
                precision, recall, f1 = eval_file(answers, dataset)
                log_line += '{}\t{}\t{}\t'.format(precision, recall, f1)
            log.write(log_line[:-1] + '\n')


def eval_negative_instances(epochs, relation='', cc=''):
    rel = lambda r: not relation or r == relation
    for epoch in epochs:
        print("----\nEPOCH: ", epoch)
        for split in ['val', 'test', 'test_hard']:
            answers = '{}{}_{}.json'.format(cc, split, epoch)
            dataset = '../Image2Synset/DATA/15_{}.tsv.tmp'.format(split)
            tp, tn, fp, fn = 0, 0, 0, 0
            with open(answers, 'r') as f, open(dataset, 'r') as d:
                ans = json.loads(f.read())
                answers = dict()
                for a in ans:
                    answers[a['question_id']] = a['answer']
                for i, line in enumerate(d.readlines()):
                    if i == 0 and i not in answers:
                        continue

                    if not rel(line.split('\t')[3]):
                        continue

                    gold = 1 if line.split('\t')[3] == 'ITSELF' else 0
                    if i not in answers:
                        print(i)
                        continue
                    if gold and answers[i]:
                        tp += 1
                    elif gold:
                        fn += 1
                    elif answers[i]:
                        fp += 1
                    else:
                        tn += 1
            precision = tp / max([(tp + fp), 0.01])
            recall = tp / max([(tp + fn), 0.01])
            f1 = 2 * precision * recall / max([(precision + recall), 0.01])
            if relation:
                print('Accuracy on {}: {}. Total examples: {}'.format(relation, tn / (fp + tn),  fp + tn))
            else:
                print("\nPrecision: ", precision)
                print("Recall: ", recall)
                print("F1: ", f1)


def main():
    # score([0, 1, 2, 3, 4, 5, 6, 7, 8])
    eval_file('imagenet_answers.json', '/home/agostina/eclipse-workspace/wsd/imagenet_dataset.tsv')
    # for cc, relation in itertools.product(['', 'cc_'], ['NOT_RELATED', 'POLYSEMY', 'SIBLING']):
    #     if not cc:
    #         print('\n\nFine-tuned, {}'.format(relation))
    #         epoch = 6
    #     else:
    #         print('\n\nPre-training, {}'.format(relation))
    #         epoch = 4
    #     eval_negative_instances([epoch], relation=relation, cc=cc)


if __name__ == '__main__':
    main()
