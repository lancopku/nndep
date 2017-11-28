'''
Python scripts for getting the statistics of the dataset, and analysis the parsing results.
'''

import sys
from collections import Counter
from argparse import ArgumentParser

ID = 0
FORM = 1
LEMMA = 2
UPOS = 3
XPOS = 4
FEATS = 5
HEAD = 6
DEPREL = 7
DEPS = 8
MISC = 9


class Entry(object):
    def __init__(self, id, form, lemma, upos, xpos, feats, head, deprel, deps,
                 misc):
        self.id = int(id)
        self.form = form
        self.lemma = lemma if lemma != '_' else None
        self.upos = upos if upos != '_' else None
        self.xpos = xpos if xpos != '_' else None
        self.feats = feats if feats != '_' else None
        self.head = int(head)
        self.deprel = deprel
        self.deps = deps if deps != '_' else None
        self.misc = misc if misc != '_' else None
        self.left_dependents = []
        self.right_dependents = []


class Sentence(object):
    def __init__(self):
        self.entries = []
        self.num_non_pu_tokens = 0
        self.num_case_of_ambiguity = 0
        self.token_involved_in_ambituity = 0
        self.num_head_initial = 0
        self.num_head_final = 0
        self.root_entry = None
        self.cnt_arc_length = Counter(
        )  # head on the left is positive (head initial) (rarc, right branch)
        self.correct_arc_length = Counter()
        self.num_correct_arcs = 0
        self.include_punct = False  # change this to include or exclude puncts

    def add_entry(self, entry):
        self.entries.append(entry)

    def __len__(self):

        return len(
            self.entries) if self.include_punct else self.num_non_pu_tokens

    def status(self):
        self.num_non_pu_tokens = len(self.entries)
        for entry in self.entries:
            if entry.upos == 'PUNCT' or entry.xpos == 'PU' or entry.xpos in [
                    ',', '.', ':', '``', "''"
            ]:
                self.num_non_pu_tokens -= 1
                if not self.include_punct:
                    continue
            if entry.head == 0:
                self.root_entry = entry.id
            else:
                self.cnt_arc_length.update([entry.id - entry.head])
                if entry.head > entry.id:
                    self.entries[entry.head - 1].left_dependents.append(
                        entry.id)
                    self.num_head_final += 1
                else:
                    self.entries[entry.head - 1].right_dependents.append(
                        entry.id)
                    self.num_head_initial += 1

        for entry in self.entries:
            if entry.left_dependents and entry.right_dependents:
                self.num_case_of_ambiguity += 1
                self.token_involved_in_ambituity += len(
                    entry.left_dependents) + len(entry.right_dependents)

    def evalulate(self, predict_sentence):
        self.correct_arc_length.clear()
        self.num_correct_arcs = 0
        for gold, pred in zip(self.entries, predict_sentence.entries):
            if not self.include_punct and (
                    gold.upos == 'PUNCT' or gold.xpos == 'PU'
                    or gold.xpos in [',', '.', ':', '``', "''"]):
                continue
            if pred.head == gold.head:
                self.num_correct_arcs += 1
                if gold.head != 0:
                    self.correct_arc_length.update([gold.id - gold.head])


class Dataset(object):
    def __init__(self):
        self.sentences = []
        self.num_forms = 0
        self.num_head_final = 0
        self.num_head_initial = 0
        self.num_case_of_ambiguity = 0
        self.num_sentence_of_ambiguity = 0
        self.num_token_in_ambiguity = 0
        self.num_token_in_ambiguity_sentence = 0
        self.cnt_arc_length = Counter()
        self.correct_arc_length = []
        self.num_correct_arcs = []

    def __len__(self):
        return len(self.sentences)

    def add_sentence(self, sentence):
        self.sentences.append(sentence)
        sentence.status()

    def evalulate(self, predict_dataset):
        #self.status()
        correct_arc_length = Counter()
        num_correct_arcs = 0
        for gold, pred in zip(self.sentences, predict_dataset.sentences):
            gold.evalulate(pred)
            num_correct_arcs += gold.num_correct_arcs
            correct_arc_length.update(gold.correct_arc_length)
        self.correct_arc_length.append(correct_arc_length)
        self.num_correct_arcs.append(num_correct_arcs)

    def status(self):
        for sentence in self.sentences:
            self.num_forms += len(sentence)
            if sentence.num_case_of_ambiguity > 0:
                self.num_sentence_of_ambiguity += 1
                self.num_token_in_ambiguity_sentence += len(sentence)
            self.num_case_of_ambiguity += sentence.num_case_of_ambiguity
            self.num_token_in_ambiguity += sentence.token_involved_in_ambituity
            self.num_head_final += sentence.num_head_final
            self.num_head_initial += sentence.num_head_initial
            self.cnt_arc_length.update(sentence.cnt_arc_length)

    def print_status(self, file=sys.stdout):
        def p(s):
            print(s, file=file)

        p('number of sentences: {}'.format(len(self)))
        p('number of forms: {}'.format(self.num_forms))
        p('average token per sentence: {:.2f}'.format(
            1.0 * self.num_forms / len(self)))
        p('')
        p('number of sentences of ambiguity: {}'.format(
            self.num_sentence_of_ambiguity))
        p('number of cases of ambiguity: {}'.format(
            self.num_case_of_ambiguity))
        p('average case of ambiguity per ambiguious sentence: {:.2f}'.format(
            1.0 * self.num_case_of_ambiguity / self.num_sentence_of_ambiguity))
        p('')
        p('number of tokens in ambiguity: {}'.format(
            self.num_token_in_ambiguity))
        p('number of tokens in ambiguious sentence: {}'.format(
            self.num_token_in_ambiguity_sentence))
        p('percentage of tokens in ambiguity: {:.2f}'.format(
            100.0 * self.num_token_in_ambiguity /
            self.num_token_in_ambiguity_sentence))
        p('')
        p('number of initial heads (right dependents): {}'.format(
            self.num_head_initial))
        p('number of final heads (left dependents): {}'.format(
            self.num_head_final))
        p('percentage of left dependents: {:.2f}'.format(
            100.0 * self.num_head_final /
            (self.num_head_final + self.num_head_initial)))
        p('')
        p('number of arcs according to direction and length:')
        for key in sorted(self.cnt_arc_length.keys()):
            p('{:>4d}:\t{}'.format(key, self.cnt_arc_length[key]))
        p('')

    def print_evaluation(self, file=sys.stdout):
        def p(s):
            print(s, file=file)

        self.print_status(file=file)

        p('uas: {}'.format('\t'.join([
            '{:.2f} ({:>4d}/{:>4d})'.format(100.0 * x / self.num_forms, x,
                                            self.num_forms)
            for x in self.num_correct_arcs
        ])))
        p('')

        for key in sorted(self.cnt_arc_length.keys()):
            num_arc = self.cnt_arc_length[key]
            #correct_arc = self.correct_arc_length[key]
            p('{:>4d}:\t{}\t{:>4d}'.format(key, '\t'.join([
                '{:>4d}'.format(counter[key])
                for counter in self.correct_arc_length
            ]), num_arc))
        p('')


def read_dataset(fp, encoding='utf-8'):
    dataset = Dataset()
    with open(fp, mode='r', encoding=encoding) as fin:
        sentence = Sentence()
        line = fin.readline()
        while line:
            if line.strip():
                if line.startswith('#'):
                    line = fin.readline()
                    continue
                columns = line.split()
                sentence.add_entry(Entry(*columns))
            else:
                dataset.add_sentence(sentence)
                sentence = Sentence()
            line = fin.readline()
    return dataset


def stat(fp, file=sys.stdout):
    dataset = read_dataset(fp)
    dataset.status()
    dataset.print_status(file=file)


def eval(fp_gold, fp_standard, fp_hybrid, file=sys.stdout):
    gold = read_dataset(fp_gold)
    hybrid = read_dataset(fp_hybrid, encoding='utf-8-sig')
    standard = read_dataset(fp_standard, encoding='utf-8-sig')
    gold.status()
    gold.evalulate(standard)
    gold.evalulate(hybrid)
    gold.print_evaluation(file=file)


with open('stats.txt', mode='w', encoding='utf-8') as fout:

    stat('trn.conllu', file=fout)
    stat('dev.conllu', file=fout)
    stat('tst.conllu', file=fout)

with open('stats-comp.txt', mode='w', encoding='utf-8') as fout:

    eval(
        'zh-ud-dev.conllu',
        'zh-ud-dev-std.conllu',
        'zh-ud-dev-hyd.conllu',
        file=fout)
