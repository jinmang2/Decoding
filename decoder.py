from abc import abstractmethod
import copy
import numpy as np
from operator import mul
import logging
from functools import reduce
from utils import Observer, Observable, MESSAGE_TYPE_DEFAULT, \
    MESSAGE_TYPE_POSTERIOR, MESSAGE_TYPE_FULL_HYPO, NEG_INF, EPS_P
import utils
from interpolation import (
    FixedInterpolationStrategy,
    EntropyInterpolationStrategy,
    MoEInterpolationStrategy
)


class Hypothesis:

    def __init__(self, trgt_sentence, total_score, score_breakdown=[]):
        self.trgt_sentence = trgt_sentence
        self.total_score = total_score
        self.score_breakdown = score_breakdown

    def __repr__(self):
        return "%s (%f)" % (' '.join(str(w) for w in self.trgt_sentence),
                            self.total_score)


class PartialHypothesis:

    def __init__(self, initial_states=None):
        self.predictor_states = initial_states
        self.trgt_sentence = []
        self.score = 0.0
        self.score_breakdown = []
        self.word_to_consume = None

    def get_last_word(self):
        if not self.trgt_sentence:
            return None
        return self.trgt_sentence[-1]

    def generate_full_hypothesis(self):
        return Hypothesis(self.trgt_sentence, self.score, self.score_breakdown)

    def _new_partial_hypo(self, states, word, score, score_breakdown):
        new_hypo = PartialHypothesis(states)
        new_hypo.score = self.score + score
        new_hypo.score_breakdown = copy.copy(self.score_breakdown)
        new_hypo.trgt_sentence = self.trgt_sentence + [word]
        new_hypo.score_breakdown.append(score_breakdown)
        return new_hypo

    def expand(self, word, new_states, score, score_breakdown):
        return self._new_partial_hypo(new_states, word, score, score_breakdown)

    def cheap_expand(self, word, score, score_breakdown):
        hypo = self._new_partial_hypo(self.predictor_states,
                                       word, score, score_breakdown)
        hypo.word_to_consume = word
        return hypo


CLOSED_VOCAB_SCORE_NORM_NONE = 1
CLOSED_VOCAB_SCORE_NORM_EXACT = 2
CLOSED_VOCAB_SCORE_NORM_REDUCED = 3
CLOSED_VOCAB_SCORE_NORM_RESCALE_UNK = 4
CLOSED_VOCAB_SCORE_NORM_NON_ZERO = 5


class  Heuristic(Observer):

    def __init__(self):
        super().__init__()
        self.predictors = []

    def set_predictors(self, predictors):
        self.predictors = predictors

    def initialize(self, src_sentence):
        pass

    @abstractmethod
    def estimate_future_cost(self, hypo):
        raise NotImplemented

    def notipy(self, message, message_type=MESSAGE_TYPE_DEFAULT):
        pass


class Decoder(Observable):

    def __init__(self, decoder_args):
        super().__init__()
        self.max_len_factor = decoder_args.max_len_factor
        self.predictors = [] # Tuples (predictor, weight)
        self.heuristics = []
        self.heuristic_predictors = []
        self.predictor_names = []
        self.allow_unk_in_output = decoder_args.allow_unk_in_output
        self.nbest = 1 # length of n-best list
        # Method Define
        self.combi_predictor_method = Decoder.combi_arithmetic_unnormalized
        self.combine_posteriors = self._combine_posteriors_norm_none
        # 기본적인 form
        self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_NONE # 1
        if decoder_args.closed_vocabulary_normalization == 'exact':
            # _combine_posteriors_with_renorm
            self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_EXACT # 2
            self.combine_posteriors = self._combine_posteriors_norm_exact
        elif decoder_args.closed_vocabulary_normalization == 'reduced':
            # _combine_posteriors_with_renorm
            self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_REDUCED # 3
            self.combine_posteriors = self._combine_posteriors_norm_reduced
        elif decoder_args.closed_vocabulary_normalization == 'rescale_unk':
            # _combine_posteriors_norm_none
            self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_RESCALE_UNK # 4
            self.combine_posteriors = self._combine_posteriors_norm_rescale_unk
        elif decoder_args.closed_vocabulary_normalization == 'non_zero':
            # _combine_posteriors_norm_none와 거의 동일!
            self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_NON_ZERO # 5
            self.combine_posteriors = self._combine_posteriors_norm_non_zero

        self.current_sen_id = -1
        self.apply_predictors_conut = 0
        self.lower_bounds = []
        if decoder_args.score_lower_bounds_file:
            with open(decoder_args.score_lower_bounds_file) as f:
                for line in f:
                    self.lower_bounds.append(float(line.strip()))
        self.interpolation_strategies = []
        self.interpolation_smoothing = decoder_args.interpolation_smoothing
        if decoding_args.interpolation_strategy:
            self.interpolaiton_mean = decoder_args.interpolation_weights_mean
            pred_strat_names = decoder_args.interpolation_strategy.split(',')
            all_strat_names = set([])
            for s in pred_strat_names:
                all_strat_names |= set(s.split("|"))
            for name in set(all_strat_names):
                pred_indices = [idx for idx, strat in enumerate(pred_strat_names)
                                    if name in strat]
                if name == 'fixed':
                    strat = FixedInterpolationStrategy()
                elif name == 'entropy':
                    strat = EntropyInterpolationStrategy(
                             decoder_args.pred_trg_vocab_size,
                             cross_entropy=False)
                elif name == 'crossentropy':
                    strat = EntropyInterpolationStrategy(
                             decoder_args.pred_trg_vocab_size,
                             cross_entropy=True)
                elif name == 'moe':
                    strat = MoEInterpolationStrategy(len(pred_indices),
                                                     decoder_args)
                else:
                    logging.error("Unknown interpolation strategy '%s'. "
                                  "Ignoring..." % name)
                    continue
                self.interpolation_strategies.append((strat, pred_indices))

    """ 얘넨 뭘까 """

    def add_predictors(self, name, predictor, weight=1.0):
        pass

    def remove_predictors(self):
        pass

    def change_predictor_weights(self, new_weights):
        pass

    def set_heuristic_predictors(self, heuristic_predictors):
        pass

    def add_heuristic(self, heuristic):
        pass

    def estimate_future_cost(self, hypo):
        pass

    def has_predictors(self):
        pass

    def consume(self, word):
        pass

    def _get_non_zero_words(self, bounded_predictors, posteriors):
        pass

    def _split_restricted_posteriors(self, predictors, posteriors):
        pass

    def apply_interpolation_strategy(
        self, pred_weights, non_zero_words, posteriors, unk_probs):
        pass

    def apply_predictors(self, top_n=0):
        pass

    """ `__init__` 메서드에서 사용될 setting """

    def _combine_posteriors_norm_none(
        self, non_zero_words, posteriors, unk_probs, pred_weights, top_n=0):
        if isinstance(non_zero_words, range) and top_n >= 0:
            non_zero_words = Decoder._scale_combine_non_zero_scores(
                len(non_zero_words), posteriors, unk_probs, pred_weights, top_n=top_n)
        combined = {}
        score_breakdown = {}
        for trgt_word in non_zero_words:
            preds = [(utils.common_get(posteriors[idx], trgt_word, unk_probs[idx]), w)
                     for idx, w in enumerate(pred_weights)]
            # Decoder.combi_arithmetic_unnormalized
            combined[trgt_word] = self.combi_predictor_method(preds)
            score_breakdown[trgt_word] = preds
        return combined, score_breakdown

    def _combine_posteriors_norm_non_zero(
        self, non_zero_words, posteriors, unk_probs, pred_weights, top_n=0):
        if isinstance(non_zero_words, range), and top_n > 0:
            non_zero_words = Decoder._scale_combine_non_zero_scores(
                len(non_zero_words), posteriors, unk_probs, pred_weights, top_n=top_n)
        combined = {}
        score_breakdown = {}
        for tgrt_word in non_zero_word:
            preds = [(utils.common_get(posteriors[idx], trgt_word, unk_probs[idx]), w)
                     for idx, w in enumerate(pred_weights)]
            # Decoder.combi_arithmetic_unnormalized
            combi_score = self.combi_predictor_method(preds)
            if abs(combi_score) <= EPS_P:
                continue
            combined[trgt_word] = combi_score
            score_breakdown[trgt_word] = preds
        return combined, score_breakdown

    @staticmethod
    def _scale_combine_non_zero_scores(
        non_zero_word_count, posteriors, unk_probs, pred_weights, top_n=0):
        scaled_posteriors = []
        for posterior, unk_prob, weight in zip(posteriors, unk_probs, pred_weights):
            if isinstance(posterior, dict):
                arr = np.full(non_zero_word_count, unk_prob)
                for word, score in posterior.items():
                    if word < non_zero_word_count:
                        arr[word] = score
                scaled_posteriors.append(arr * weight)
            else:
                n_unks = non_zero_word_count - len(posterior)
                if n_unks > 0:
                    posterior = np.concatenate((posterior, np.full(n_unks, unk_prob)))
                elif n_unks < 0:
                    posterior = posterior[:n_unks]
                scaled_posteriors.append(posterior * weight)
        combined_scores = np.sum(scaled_posteriors, axis=0)
        return utils.argmax_n(combined_scores, top_n)

    def _combine_posteriors_with_renorm(self, score_breakdown_raw, renorm_factors):
        """ Helper function for ``_combine_posteriors_norm_*`` functions """
        n_predictors = len(self.predictors)
        combined = {}
        score_breakdown = {}
        for trgt_word, preds_raw in score_breakdown_raw.items():
            preds = [(pred_raw[idx][0] - renorm_factors[idx], preds_raw[idx][1])
                     for idx in range(n_predictors)]
            # Decoder.combi_arithmetic_unnormalized
            combined[trgt_word] = self.combi_predictor_method(preds)
            score_breakdown[trgt_word] = preds
        return combined, score_breakdown

    def _combine_posteriors_norm_exact(
        self, non_zero_words, posteriors, unk_probs, pred_weights, top_n=0):
        n_predictors = len(self.predictors)
        score_breakdown_raw = {}
        unk_counts = [0] * n_predictors
        for trgt_word in non_zero_words:
            preds = []
            for idx, w in enumerate(pred_weights):
                if utils.common_contains(posteriors[idx], trgt_word):
                    preds.append((posteriors[idx][trgt_word], w))
                else:
                    preds.append((unk_probs[idx], w))
                    unk_counts[idx] += 1
            score_breakdown_raw[trgt_word] = preds
        renorm_factors = [0.0] * n_predictors
        for idx in range(n_predictors):
            if unk_counts[idx] > 1:
                renorm_factors[idx] = np.log(
                    1.0 + (unk_counts[idx] - 1.0) * np.exp(unk_probs[idx]))
        return self._combine_posteriors_with_renorm(
            score_breakdown_raw, renorm_factors)


    def _combine_posteriors_norm_reduced(
        self, non_zero_words, posteriors, unk_probs, pred_weights, top_n=0):
        n_predictors = len(self.predictors)
        score_breakdown_raw = {}
        for trgt_word  in non_zero_word:
            score_breakdown_raw[trgt_word] = [
                (utils.common_get(posteriors[idx], trgt_word, unk_probs[idx]), w)
                for idx, w in enumerate(pred_weights)]
        sums = []
        for idx in range(n_predictors):
            sums.append( utils.log_sum(
                    [preds[idx][0] for preds in score_breakdown_raw.values()] ) )
        return self._combine_posteriors_with_renorm(score_breakdown_raw, sums)

    def _combine_posteriors_norm_rescale_unk(
        self, non_zero_words, posteriors, unk_probs, pred_weights, top_n=0):
        n_predictors = len(self.predictors)
        unk_counts = [0.0] * n_predictors
        for idx, w in enumerate(pred_weights):
            if unk_probs[idx] >= EPS_P or unk_probs[idx] == NEG_INF:
                continue
            for trgt_word in non_zero_words:
                if not utils.common_contains(posteriors[idx], trgt_word):
                    unk_counts[idx] += 1.0
        return self._combine_posteriors_norm_none(
            non_zero_words,
            posteriors,
            [unk_probs[idx] - np.log(max(1.0, unk_counts[idx]))
             for idx in range(n_predictors)],
             top_n
        )

    @staticmethod
    def combi_arithmetic_unnormalized(x):
        # return sum(f*w for f, w in x)
        (fAcc, _) = reduce(lambda x1, x2: (x1[0]*x1[1] + x2[0]*x2[1], 1.0), x, (0.0, 1.0))
        return fAcc

    """ Others """
    def set_current_sen_id(self, sen_id):
        pass

    def initialize_predictors(self, src_sentence):
        pass

    def add_full_hypo(self, hypo):
        pass

    def get_full_hypos_sorted(self):
        pass

    def get_lower_score_bound(self):
        pass

    def get_max_expansions(self, max_expansions_params, src_sentence):
        pass

    def set_predictor_states(self, states):
        pass

    def get_predictors_states(self):
        pass

    """ Core method """
    @abstractmethod
    def decode(self, src_sentence):
        """
        단일 source sentence를 디코딩하는 메서드이며 이는 subclass에서 반드시 구현되어야 함.
        Search 전략 구현의 핵심을 담고 있으며 ``src_sentence``는 <S>나 </S> 심볼을 제외한
        source sentence를 표현하는 source 단어 id들의 리스트이다.
        메서드는 가설들(hypotheses)의 리스트를 반환하며
        """
