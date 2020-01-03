import cv2
import numpy as np
import os


def remove_filtered_images(batch):
    with open('silver_batch_{}_filtered.tsv'.format(batch), 'r') as f, open('silver_batch_{}.tsv'.format(batch), 'r') as old:
        f_lines = f.readlines()
        old_lines = old.readlines()
        i = 0
        missing = 0
        for line in old_lines:
            if f_lines[i] == line:
                i += 1
            elif os.path.isfile(line.split()[0]):
                missing += 1
    print(missing)


for batch in range(5):
    with open('silver_batch_{}.tsv'.format(batch), 'r') as f, open('silver_batch_{}_filtered.tsv'.format(batch), 'w+') as w:
        for line in f.readlines():
            im = cv2.imread(line.split()[0])
            try:
                im = im.astype(np.float32, copy=False)
                a = w.write(line)
            except Exception as e:
                pass
                # print(e)