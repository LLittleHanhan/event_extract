from var import train_path, dev_path, test_path

with open(test_path, 'r', encoding='utf-8') as f, open('./class/new_test.txt', 'w', encoding='utf-8') as nf:
    for l in f.readlines():
        title, idx = l.strip().split('\t')
        if int(idx) != 1:
            if int(idx) > 1:
                idx = str(int(idx)-1)
            nf.write(title + '\t' + idx + '\n')
