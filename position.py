pos_labels = ['OFF', 'ON']
pos_label_dict = {l: i for i, l in enumerate(pos_labels)}
POS_NUM = len(pos_labels)


def get_index(label):
    return pos_label_dict[label]


def get_label(index):
    return pos_labels[index]


def compute_reward(pos_idx, speed):
    return -0.2 + speed / 28 if pos_idx else -0.3 - speed / 40


if __name__ == '__main__':
    print(get_index('ON'))
    print(get_index('OFF'))

    print(get_label(0))
    print(get_label(1))
