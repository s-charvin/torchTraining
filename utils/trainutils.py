import torch


def collate_fn_pad(batch):

    # 一个 batch 大小的列表，内部含有数据元组，如：(waveform, label)
    # Gather in lists

    tensors = []
    genders = []
    labels = []
    nfs = None

    for spec, gender, label in batch:
        tensors.append(spec)
        genders.append(gender)
        labels.append(label)
        # texts.append(texts)
    tensors_lengths = [i.shape[0] for i in tensors]
    tensors = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.)

    packed_tensors = torch.nn.utils.rnn.pack_padded_sequence(
        tensors, tensors_lengths, batch_first=True, enforce_sorted=False)

    return packed_tensors, genders, torch.LongTensor(labels), nfs


def collate_fn_split(batch):
    fix_len = 260
    hop_len = 130

    # 一个batch大小的列表，内部含有数据元组，如：()
    # Gather in lists
    tensors = []
    genders = []
    labels = []
    # texts = []
    nfs = []
    for spec, gender, label in batch:
        if len(spec.shape) == 2:
            # item, label: [frame, feature] , int
            # frame 分段为固定大小, 125代表采样率为16000, 帧移为256条件下, 2秒时间长度的数据
            # 计算总分段数
            if spec.shape[0] < fix_len - hop_len:
                nf = 1
            else:
                nf = (spec.shape[0] - fix_len + hop_len) // hop_len + 1
            nfs.append(nf)
            # 计算每段的起始位置
            indf = [i*hop_len for i in range(nf)]
            # 正式开始分段数据
            for i in range(nf-1):
                tensors.append(
                    spec[indf[i]:indf[i] + fix_len, :])
                genders.append(gender)
                labels.append(label)
            tensors.append(spec[indf[-1]:, :])
            genders.append(gender)
            labels.append(label)

        elif len(spec.shape) == 1:
            fix_len = 16000 * 4
            hop_len = 16000 * 2
            if spec.shape[0] < fix_len - hop_len:
                nf = 1
            else:
                nf = (spec.shape[0] - fix_len + hop_len) // hop_len + 1
            nfs.append(nf)
            # 计算每段的起始位置
            indf = [i*hop_len for i in range(nf)]
            # 正式开始分段数据
            for i in range(nf-1):
                tensors.append(
                    spec[indf[i]:indf[i] + fix_len])
                genders.append(gender)
                labels.append(label)
            tensors.append(spec[indf[-1]:])
            genders.append(gender)
            labels.append(label)
            # 对不等长的数据进行填充
    tensors_lengths = [i.shape[0] for i in tensors]
    tensors = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0.)
    if tensors.shape[1] != fix_len:
        print(tensors.shape[1])
        tensors = torch.nn.functional.pad(
            tensors, (0, 0, 0, fix_len-tensors.shape[1], 0, 0), mode='constant', value=0.)
    packed_tensors = torch.nn.utils.rnn.pack_padded_sequence(
        tensors, tensors_lengths, batch_first=True, enforce_sorted=False)

    return packed_tensors, genders, torch.LongTensor(labels), nfs
