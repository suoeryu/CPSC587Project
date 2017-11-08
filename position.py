pos_labels = ['OFF', 'ON']
pos_label_dict = {l: i for i, l in enumerate(pos_labels)}
POS_NUM = len(pos_labels)


def get_index(label):
    return pos_label_dict[label]


def get_label(index):
    return pos_labels[index]


if __name__ == '__main__':
    print(get_index('ON'))
    print(get_index('OFF'))

    print(get_label(0))
    print(get_label(1))
