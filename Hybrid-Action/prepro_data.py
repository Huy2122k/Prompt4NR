import re
import random
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
import torch


class MyDataset(Dataset):
    def __init__(self, args, tokenizer, news_dict, conti_tokens, status='train'):
        self.tokenizer = tokenizer
        self.news_dict = news_dict
        self.args = args
        self.status = status
        self.conti_tokens = conti_tokens

        self.data = []
        self.imp_lens = []
        # Tiền biên dịch biểu thức chính quy để làm sạch văn bản
        self._pattern = re.compile(r'[^A-Za-z0-9 ]+')

        # Chọn file dữ liệu dựa trên status
        if self.status == 'train':
            self.data_path = os.path.join(args.train_data_path, 'train.txt')
        elif self.status == 'val':
            self.data_path = os.path.join(args.data_path, 'val.txt')
        else:
            self.data_path = os.path.join(args.data_path, 'test.txt')
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def obtain_data(self, data):
        # Trả về 4 phần của data. (Có thể giới hạn kích thước cho debug nếu cần)
        return data[0], data[1], data[2], data[3]

    def _process_title(self, title: str, max_len: int) -> str:
        """Làm sạch và cắt tiêu đề về số từ tối đa."""
        cleaned = self._pattern.sub('', title)
        tokens = cleaned.split()
        return ' '.join(tokens[:max_len])

    def _generate_history_sentence(self, his_clicks, max_title_len: int, max_his_len: int) -> str:
        """Tạo câu lịch sử của người dùng từ danh sách tin đã click."""
        # Lấy tối đa max_his tin click và đảo thứ tự (tin mới nhất đầu tiên)
        his_clicks = his_clicks[-self.args.max_his:]
        his_clicks = list(reversed(his_clicks))
        his_titles = [
            self._process_title(self.news_dict[news]['title'], max_title_len)
            for news in his_clicks
        ]
        his_sen = '[NSEP] ' + ' [NSEP] '.join(his_titles)
        his_sen_ids = self.tokenizer.encode(his_sen, add_special_tokens=False)[:max_his_len]
        return self.tokenizer.decode(his_sen_ids)

    def prepro_train(self, imp_ids, behaviors, news_dict, K_samples,
                     max_his=50, max_title_len=10, max_candi_len=20, max_his_len=450):
        # Tạo template cơ bản một lần
        template1 = ''.join(self.conti_tokens[0]) + "<user_sentence>"
        template2 = ''.join(self.conti_tokens[1]) + "<candidate_news>"
        template3 = "Does the user click the news? [MASK]"
        template = template1 + "[SEP]" + template2 + "[SEP]" + template3

        for impid, behav in zip(imp_ids, behaviors):
            # Tạo câu lịch sử từ danh sách tin đã click
            his_sen = self._generate_history_sentence(behav[0], max_title_len, max_his_len)
            base_sentence = template.replace("<user_sentence>", his_sen)

            positives = behav[1]
            negatives = behav[2]

            for pos_news in positives:
                pos_title = self._process_title(news_dict[pos_news]['title'], max_candi_len)
                sentence = base_sentence.replace("<candidate_news>", pos_title)
                self.data.append({'sentence': sentence, 'target': 1, 'imp': impid})

                # Lấy mẫu ngẫu nhiên K negatives cho mỗi positive
                if len(negatives) >= K_samples:
                    sample_negs = random.sample(negatives, k=K_samples)
                else:
                    sample_negs = np.random.choice(negatives, K_samples, replace=True).tolist()

                for neg in sample_negs:
                    neg_title = self._process_title(news_dict[neg]['title'], max_candi_len)
                    neg_sentence = base_sentence.replace("<candidate_news>", neg_title)
                    self.data.append({'sentence': neg_sentence, 'target': 0, 'imp': impid})

    def prepro_dev(self, imp_ids, behaviors, news_dict,
                   max_his=50, max_title_len=10, max_candi_len=20, max_his_len=450):
        # Tạo template cơ bản một lần
        template1 = ''.join(self.conti_tokens[0]) + "<user_sentence>"
        template2 = ''.join(self.conti_tokens[1]) + "<candidate_news>"
        template3 = "Does the user click the news? [MASK]"
        template = template1 + "[SEP]" + template2 + "[SEP]" + template3

        for impid, behav in zip(imp_ids, behaviors):
            # Nếu không có tin click thì bỏ qua
            if not behav[0]:
                continue
            his_sen = self._generate_history_sentence(behav[0], max_title_len, max_his_len)
            base_sentence = template.replace("<user_sentence>", his_sen)

            positives = behav[1]
            negatives = behav[2]
            for pos_news in positives:
                pos_title = self._process_title(news_dict[pos_news]['title'], max_candi_len)
                sentence = base_sentence.replace("<candidate_news>", pos_title)
                self.data.append({'sentence': sentence, 'target': 1, 'imp': impid})

            for neg in negatives:
                neg_title = self._process_title(news_dict[neg]['title'], max_candi_len)
                sentence = base_sentence.replace("<candidate_news>", neg_title)
                self.data.append({'sentence': sentence, 'target': 0, 'imp': impid})

    def load_data(self):
        # Sử dụng context manager để mở file đảm bảo đóng file đúng cách
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        imps, users, times, behaviors = self.obtain_data(data)
        if self.status == 'train':
            self.prepro_train(
                imps, behaviors, self.news_dict,
                self.args.num_negs,
                max_his=self.args.max_his,
                max_title_len=getattr(self.args, 'max_title_len', 10),
                max_candi_len=getattr(self.args, 'max_candi_len', 20),
                max_his_len=self.args.max_his_len
            )
        else:
            self.prepro_dev(
                imps, behaviors, self.news_dict,
                max_his=self.args.max_his,
                max_title_len=getattr(self.args, 'max_title_len', 10),
                max_candi_len=getattr(self.args, 'max_candi_len', 20),
                max_his_len=self.args.max_his_len
            )

    def collate_fn(self, batch):
        sentences = [x['sentence'] for x in batch]
        target = [x['target'] for x in batch]
        imp_id = [x['imp'] for x in batch]

        encode_dict = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.args.max_tokens,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        batch_enc = encode_dict['input_ids']
        batch_attn = encode_dict['attention_mask']
        target = torch.LongTensor(target)

        return batch_enc, batch_attn, target, imp_id
