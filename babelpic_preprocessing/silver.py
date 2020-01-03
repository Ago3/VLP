def batch_stats(img_names_file, batch_size, batch_number, start=0):
    with open(img_names_file, 'r') as f, open('silver_batch_{}.tsv'.format(batch_number), 'w+') as w:
        synsets = set()
        instances = 0
        lines = f.readlines()
        for i, line in enumerate(lines[start:]):
            synset = line.split('_')[0]
            synsets.add(synset)
            if len(synsets) > batch_size:
                break
            instances += 1
            w.write(line)
    print("Instances: {}".format(instances))
    return start + i


def generate_all_batches(img_names_file, batch_size):
    start = 0
    with open(img_names_file, 'r') as f:
        size = len(f.readlines())
        batch_number = 0
        while start < size - 1:
            start = batch_stats(img_names_file, batch_size, batch_number, start)
            batch_number += 1


def main():
    img_names_file = 'img_names_silver.tsv'
    generate_all_batches(img_names_file, 9500)


if __name__ == '__main__':
    main()
