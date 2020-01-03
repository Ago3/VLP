import numpy as np
import torch


def main():
    d = np.load('sensemb.npy', allow_pickle=True)
    synsets = [list(x.keys())[0] for x in d]
    matrix = torch.stack([list(x.values())[0].cpu() for x in d], 0)
    norm_matrix = matrix / ((matrix ** 2).sum(1) ** 0.5).view(-1, 1)
    similarities = norm_matrix @ norm_matrix.t()
    with open('neighbours.tsv', 'w+') as f:
        for i, synset in enumerate(synsets):
            topk = similarities[i].topk(11)[1][1:]
            topk_synsets = [synsets[i] for i in topk]
            f.write(synset + '\t' + '\t'.join(topk_synsets) + '\n')


if __name__ == '__main__':
    main()
