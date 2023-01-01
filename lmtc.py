import random
from torch.utils import data
import torch
from torch import nn
import os
import logging
import json
import re
import time
import tempfile
import glob
import tqdm
import numpy as np
from copy import deepcopy
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_score, recall_score
from json_loader import JSONLoader
from transformers import AutoTokenizer
from data import DATA_SET_DIR, MODELS_DIR
from configuration import Configuration
from metrics import mean_recall_k, mean_precision_k, mean_ndcg_score, mean_rprecision_k
from model import Model
# import warnings
# warnings.filterwarnings("ignore")
LOGGER = logging.getLogger(__name__)
class LMTC:
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(Configuration['model']['bert'])
        self.load_label_descriptions()
    def load_label_descriptions(self):
        LOGGER.info('Load labels\' data')
        LOGGER.info('-------------------')

        # Load train dataset and count labels
        train_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'train', '*.json'))
        train_counts = Counter()
        for filename in tqdm.tqdm(train_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    train_counts[concept] += 1

        train_concepts = set(list(train_counts))#存训练集中所有标签id

        frequent, few = [], []#分别存训练集标签id
        for i, (label, count) in enumerate(train_counts.items()):
            if count > Configuration['sampling']['few_threshold']:
                frequent.append(label)
            else:
                few.append(label)

        # Load dev/test datasets and count labels
        rest_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'dev', '*.json'))
        rest_files += glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'test', '*.json'))
        rest_concepts = set()#dev test中的标签id
        for filename in tqdm.tqdm(rest_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    rest_concepts.add(concept)

        # Load label descriptors
        with open(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'],
                               '{}.json'.format(Configuration['task']['dataset']))) as file:
            data = json.load(file)
            none = set(data.keys())

        none = none.difference(train_concepts.union((rest_concepts)))#dev和test以及train集合中的所有标签与none作差集得到在none中有的但不在数据集里的
        parents = []#获得父母标签
        for key, value in data.items():
            parents.extend(value['parents'])
        none = none.intersection(set(parents))#再与所有的父母标签作交集得到 没出现在dev test train集中的标签

        # Compute zero-shot group
        zero = list(rest_concepts.difference(train_concepts))#出现在test和dev里但是没出现在训练集中的
        true_zero = deepcopy(zero)#浅拷贝一份
        zero = zero + list(none)

        self.label_ids = dict()#4654
        self.margins = [(0, len(frequent) + len(few) + len(true_zero))]#[(0,总长度)]
        k = 0
        for group in [frequent, few, zero]:
            self.margins.append((k, k + len(group)))#[(0,len(frequent)),(len(frequent),len(frequnet) + len(few)),(len,)]
            for concept in group:
                self.label_ids[concept] = k#frequnt、few、zero从0开始标记 包含了未出现过的母亲节点
                k += 1
        self.margins[-1] = (self.margins[-1][0], len(frequent) + len(few) + len(true_zero))#真正的值

        label_terms = []#[['international', 'affairs'],...]存储解释器里所有的label对应的值
        for i, (label, index) in enumerate(self.label_ids.items()):
            label_terms.append([token for token in word_tokenize(data[label]['label']) if re.search('[A-Za-z]', token)])


        LOGGER.info('#Labels:         {}'.format(len(label_terms)))
        LOGGER.info('Frequent labels: {}'.format(len(frequent)))
        LOGGER.info('Few labels:      {}'.format(len(few)))
        LOGGER.info('Zero labels:     {}'.format(len(true_zero)))

    def load_dataset(self, dataset_name):
        """
        Load dataset and return list of documents 将json转换为list
        :param dataset_name: the name of the dataset
        :return: list of Document objects
        """
        filenames = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], dataset_name, '*.json'))
        loader = JSONLoader()

        documents = []
        for filename in tqdm.tqdm(sorted(filenames)):
            documents.append(loader.read_file(filename))

        return documents#返回的是一个列表 元素为document对象

    def process_dataset(self, documents):
        """
         Process dataset documents (samples) and create targets
         :param documents: list of Document objects
         :return: samples, targets
         """
        samples = []
        targets = []
        for document in documents:
            samples.append(document.tokens)
            targets.append(document.tags)
         

        del documents
        return samples, targets#list list

    def encode_dataset(self, sequences, tags):
        temp = [' '.join(seq) for seq in sequences]#32
        samples = self.tokenizer.batch_encode_plus(temp,padding=True,truncation=True,max_length=512,return_tensors="pt")
        targets = torch.zeros((len(sequences), len(self.label_ids)), dtype=torch.float32)#tensor
        for i, (document_tags) in enumerate(tags):
            for tag in document_tags:
                if tag in self.label_ids:
                    targets[i][self.label_ids[tag]] = 1.
        del sequences, tags
        return samples['input_ids'], targets

    def data_iter(self, samples, targets, batch_size=32):
        x, y = self.encode_dataset(samples, targets)
        x, y = x.cuda(), y.cuda()
        dataset = data.TensorDataset(x, y)
        return data.DataLoader(dataset, batch_size, shuffle=True)#youke

    def calculate_performance(self, model, generator):
        true_tmp = torch.ones((1,4654))#最后需忽略此项
        pred_tmp = torch.ones((1,4654))#最后需忽略此项
        with torch.no_grad():
            for X, y in generator:
                y_hat = model(X)
                y_hat = y_hat.cpu()
                y = y.cpu()
                true_tmp = torch.cat((true_tmp,y),dim=0)
                pred_tmp = torch.cat((pred_tmp,y_hat),dim=0)

        predictions = pred_tmp[1:,:]#忽略第一行
        true_targets = true_tmp[1:,:]#忽略第一行

        pred = torch.where(predictions > 0.5,1.0,0.0)
        predictions = predictions.numpy()
        pred_targets = pred.numpy()#转numpy
        true_targets = true_targets.numpy()
        template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'
        # Overall
        for labels_range, frequency, message in zip(self.margins,
                                                    ['Overall', 'Frequent', 'Few', 'Zero'],
                                                    ['Overall', 'Frequent Labels (>=50 Occurrences in train set)',
                                                     'Few-shot (<=50 Occurrences in train set)',
                                                     'Zero-shot (No Occurrences in train set)']):
            start, end = labels_range
            LOGGER.info(message)
            LOGGER.info('----------------------------------------------------')
            for average_type in ['micro', 'macro', 'weighted']:
                p = precision_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                r = recall_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                f1 = f1_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))

            for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                r_k = mean_recall_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                p_k = mean_precision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                rp_k = mean_rprecision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                ndcg_k = mean_ndcg_score(true_targets[:, start:end], predictions[:, start:end], k=i)
                LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
            LOGGER.info('----------------------------------------------------')

    def train(self):

        LOGGER.info('\n---------------- Train Starting ----------------')

        for param_name, value in Configuration['model'].items():
            LOGGER.info('\t{}: {}'.format(param_name, value))

        # Load training/validation data
        LOGGER.info('Load training/validation data')
        LOGGER.info('------------------------------')

        documents = self.load_dataset('train')
        train_samples, train_tags = self.process_dataset(documents)
        train_generator = self.data_iter(train_samples, train_tags, batch_size=Configuration['model']['batch_size'])

        documents = self.load_dataset('dev')
        val_samples, val_tags = self.process_dataset(documents)
        val_generator = self.data_iter(val_samples, val_tags, batch_size=Configuration['model']['batch_size'])

        # Compile neural network
        LOGGER.info('Compile neural network')
        LOGGER.info('------------------------------')
        net = Model(n_classes=4654,dropout_rate=0.5)
        net = net.cuda()
        #Adam
        optimizer = torch.optim.Adam(net.parameters(), lr=Configuration['model']['lr'])
        #二分类损失函数
        #loss = torch.nn.BCELoss()
        loss = torch.nn.functional.binary_cross_entropy
        save_path = os.path.join(MODELS_DIR, '{}_{}.h5'.format(Configuration['task']['dataset'].upper(),
                                                         Configuration['model']['architecture'].upper()))# 当前目录下
        LOGGER.info('Training model')
        LOGGER.info('-----------')
        start_time = time.time()
        loss_val_mean = 1
        for epoch in range(Configuration['model']['epochs']):
            net.train()
            loss_sum = 0
            number = 0
            with tqdm.tqdm(train_generator,unit="batch") as loop:
                for X,y in loop:
                    number += 1
                    optimizer.zero_grad()
                    y_hat = net(X)
                    l = loss(y_hat,y)
                    loss_sum += l
                    optimizer.zero_grad()
                    l.mean().backward()
                    optimizer.step()
                    loop.set_description(f'Epoch [{epoch + 1}/{20}]')
                    loop.set_postfix(loss=loss_sum/number)
            with torch.no_grad():
                loss_val = 0
                number_val = 0
                net.eval()  # 将模型改为预测模式
                for features, label in val_generator:
                    number_val += 1
                    out = net(features)  # 经网络输出的结果
                    l2 = loss(out, label)  # 得到误差
                    # 记录误差
                    loss_val += l2
                tmp = loss_val / number_val
                # if loss_val_mean > tmp:
                #     loss_val_mean = tmp
                #     torch.save(net, save_path)
                print(f'eval_loss {float(loss_val / number_val):f}')


        LOGGER.info('---------model is saved-------------')
        torch.save(net, save_path)
        LOGGER.info('---------train is end-------------')
        #del train_generator
        #读取模型
        LOGGER.info('---------load model-------------')
        model = torch.load(save_path)

        LOGGER.info('Load valid data')
        LOGGER.info('------------------------------')
        self.calculate_performance(model=model, generator=val_generator)

        LOGGER.info('Load test data')
        LOGGER.info('------------------------------')
        test_documents = self.load_dataset('test')
        test_samples, test_tags = self.process_dataset(test_documents)
        test_generator = self.data_iter(test_samples, test_tags,batch_size=Configuration['model']['batch_size'])
        self.calculate_performance(model=model, generator=test_generator)

        total_time = time.time() - start_time
        LOGGER.info('\nTotal Training Time: {} secs'.format(total_time))

if __name__ == '__main__':
    Configuration.configure()
    LMTC().train()

