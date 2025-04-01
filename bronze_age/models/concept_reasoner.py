import abc
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from bronze_age.config import BronzeConfig, Config
from bronze_age.models.util import DifferentiableArgmax


class Logic:
    @abc.abstractmethod
    def update(self):
        raise NotImplementedError

    @abc.abstractmethod
    def conj(self, a, dim=1):
        raise NotImplementedError

    @abc.abstractmethod
    def disj(self, a, dim=1):
        raise NotImplementedError

    def conj_pair(self, a, b):
        raise NotImplementedError

    def disj_pair(self, a, b):
        raise NotImplementedError

    def iff_pair(self, a, b):
        raise NotImplementedError

    @abc.abstractmethod
    def neg(self, a):
        raise NotImplementedError


class ProductTNorm(Logic):
    def __init__(self):
        super(ProductTNorm, self).__init__()
        self.current_truth = torch.tensor(1)
        self.current_false = torch.tensor(0)

    def update(self):
        pass

    def conj(self, a, dim=1):
        return torch.prod(a, dim=dim, keepdim=True)

    def conj_pair(self, a, b):
        return a * b

    def disj(self, a, dim=1):
        return 1 - torch.prod(1 - a, dim=dim, keepdim=True)

    def disj_pair(self, a, b):
        return a + b - a * b

    def iff_pair(self, a, b):
        return self.conj_pair(
            self.disj_pair(self.neg(a), b), self.disj_pair(a, self.neg(b))
        )

    def neg(self, a):
        return 1 - a

    def predict_proba(self, a):
        return a.squeeze(-1)


class GodelTNorm(Logic):
    def __init__(self):
        super(GodelTNorm, self).__init__()
        self.current_truth = 1
        self.current_false = 0

    def update(self):
        pass

    def conj(self, a, dim=1):
        return torch.min(a, dim=dim, keepdim=True)[0]

    def disj(self, a, dim=1):
        return torch.max(a, dim=dim, keepdim=True)[0]

    def conj_pair(self, a, b):
        return torch.minimum(a, b)

    def disj_pair(self, a, b):
        return torch.maximum(a, b)

    def iff_pair(self, a, b):
        return self.conj_pair(
            self.disj_pair(self.neg(a), b), self.disj_pair(a, self.neg(b))
        )

    def neg(self, a):
        return 1 - a

    def predict_proba(self, a):
        return a.squeeze(-1)


def softselect(values, temperature):
    softmax_scores = torch.log_softmax(values, dim=1)
    softscores = torch.sigmoid(
        softmax_scores - temperature * softmax_scores.mean(dim=1, keepdim=True)
    )
    return softscores


class ConceptReasoningLayer(torch.nn.Module):
    def __init__(
        self,
        emb_size,
        n_classes,
        logic: Logic = GodelTNorm(),
        temperature: float = 0.5,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.logic: Logic = logic
        self.filter_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.sign_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.temperature = temperature

    def forward(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)

        if sign_attn is None:
            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence
            sign_attn = torch.sigmoid(self.sign_nn(x))

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)
        sign_terms = self.logic.iff_pair(sign_attn, values)

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
            filter_attn = softselect(self.filter_nn(x), self.temperature)

        # filter value
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(sign_terms, self.logic.neg(filter_attn))

        # generate minterm
        preds = self.logic.conj(filtered_values, dim=1).squeeze(1).float()

        if return_attn:
            return preds, sign_attn, filter_attn
        else:
            return preds

    def explain(
        self, x, c, mode, concept_names=None, class_names=None, filter_attn=None
    ):
        assert mode in ["local", "global", "exact"]

        if concept_names is None:
            concept_names = [f"c_{i}" for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f"y_{i}" for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        y_preds, sign_attn_mask, filter_attn_mask = self.forward(
            x, c, return_attn=True, filter_attn=filter_attn
        )

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(x)):
            prediction = y_preds[sample_idx] > 0.5
            active_classes = torch.argwhere(prediction).ravel()

            if len(active_classes) == 0:
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append(
                    {
                        "class": -1,
                        "explanation": "",
                        "attention": [],
                    }
                )
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = []
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        sign_attn = sign_attn_mask[
                            sample_idx, concept_idx, target_class
                        ]
                        filter_attn = filter_attn_mask[
                            sample_idx, concept_idx, target_class
                        ]

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the filter attention score is lower than the concept probability
                        at_score = 0
                        sign_terms = self.logic.iff_pair(sign_attn, c_pred).item()
                        if self.logic.neg(filter_attn) < sign_terms:
                            if sign_attn >= 0.5:
                                # if the concept is relevant and the sign is positive we just take its attention score
                                at_score = filter_attn.item()
                                if mode == "exact":
                                    minterm.append(
                                        f"{sign_terms:.3f} ({concept_names[concept_idx]})"
                                    )
                                else:
                                    minterm.append(f"{concept_names[concept_idx]}")
                            else:
                                # if the concept is relevant and the sign is positive we take (-1) * its attention score
                                at_score = -filter_attn.item()
                                if mode == "exact":
                                    minterm.append(
                                        f"{sign_terms:.3f} (~{concept_names[concept_idx]})"
                                    )
                                else:
                                    minterm.append(f"~{concept_names[concept_idx]}")
                        attentions.append(at_score)

                    # add explanation to list
                    target_class_name = class_names[target_class]
                    minterm = " & ".join(minterm)
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append(
                        {
                            "sample-id": sample_idx,
                            "class": target_class_name,
                            "explanation": minterm,
                            "attention": attentions,
                        }
                    )

        if mode == "global":
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append(
                        {
                            "class": class_id,
                            "explanation": explanation,
                            "count": count,
                        }
                    )

        return explanations


class GlobalConceptReasoningLayer(nn.Module):
    def __init__(
        self,
        n_concepts,
        n_classes,
        logic: Logic = GodelTNorm(),
        temperature: float = 0.5,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.logic: Logic = logic
        self.filter_nn = nn.Embedding(n_concepts, n_classes)
        self.sign_nn = nn.Embedding(n_concepts, n_classes)
        self.temperature = temperature

    def forward(self, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)

        if sign_attn is None:
            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence
            sign_attn = torch.sigmoid(self.sign_nn.weight[None, ...])

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)
        sign_terms = self.logic.iff_pair(sign_attn, values)

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
            filter_attn = softselect(self.filter_nn.weight[None, ...], self.temperature)

        # filter value
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(sign_terms, self.logic.neg(filter_attn))

        # generate minterm
        preds = self.logic.conj(filtered_values, dim=1).squeeze(1).float()

        if return_attn:
            return preds, sign_attn, filter_attn
        else:
            return preds

    def explain(self, c, mode, concept_names=None, class_names=None, filter_attn=None):
        assert mode in ["local", "global", "exact"]

        if concept_names is None:
            concept_names = [f"c_{i}" for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f"y_{i}" for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        y_preds, sign_attn_mask, filter_attn_mask = self.forward(
            c, return_attn=True, filter_attn=filter_attn
        )
        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(c)):
            prediction = y_preds[sample_idx] > 0.5
            active_classes = torch.argwhere(prediction).ravel()

            if len(active_classes) == 0:
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append(
                    {
                        "class": -1,
                        "explanation": "",
                        "attention": [],
                    }
                )
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = []
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        sign_attn = sign_attn_mask[0, concept_idx, target_class]
                        filter_attn = filter_attn_mask[0, concept_idx, target_class]

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the filter attention score is lower than the concept probability
                        at_score = 0
                        sign_terms = self.logic.iff_pair(sign_attn, c_pred).item()
                        if self.logic.neg(filter_attn) < sign_terms:
                            if sign_attn >= 0.5:
                                # if the concept is relevant and the sign is positive we just take its attention score
                                at_score = filter_attn.item()
                                if mode == "exact":
                                    minterm.append(
                                        f"{sign_terms:.3f} ({concept_names[concept_idx]})"
                                    )
                                else:
                                    minterm.append(f"{concept_names[concept_idx]}")
                            else:
                                # if the concept is relevant and the sign is positive we take (-1) * its attention score
                                at_score = -filter_attn.item()
                                if mode == "exact":
                                    minterm.append(
                                        f"{sign_terms:.3f} (~{concept_names[concept_idx]})"
                                    )
                                else:
                                    minterm.append(f"~{concept_names[concept_idx]}")
                        attentions.append(at_score)

                    # add explanation to list
                    target_class_name = class_names[target_class]
                    minterm = " & ".join(minterm)
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append(
                        {
                            "sample-id": sample_idx,
                            "class": target_class_name,
                            "explanation": minterm,
                            "attention": attentions,
                        }
                    )

        if mode == "global":
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append(
                        {
                            "class": class_id,
                            "explanation": explanation,
                            "count": count,
                        }
                    )

        return explanations


class ConceptReasonerModule(torch.nn.Module):
    def __init__(self, n_concepts, n_classes, emb_size, config: Config | BronzeConfig):
        super(ConceptReasonerModule, self).__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.n_concepts = n_concepts
        self.concept_reasoner = ConceptReasoningLayer(
            emb_size=emb_size,
            n_classes=n_classes,
            temperature=config.concept_temperature,
        )
        self.concept_context_generator = torch.nn.Sequential(
            torch.nn.Linear(self.n_concepts, 2 * emb_size * self.n_concepts),
            torch.nn.LeakyReLU(),
        )

        self.diff_argmax = DifferentiableArgmax(config)

    def forward(self, combined, return_explanation=False, concept_names=None):
        concept_embs = self.concept_context_generator(combined)
        concept_embs_shape = combined.shape[:-1] + (self.n_concepts, 2 * self.emb_size)
        concept_embs = concept_embs.view(*concept_embs_shape)
        concept_pos = concept_embs[..., : self.emb_size]
        concept_neg = concept_embs[..., self.emb_size :]
        embedding = concept_pos * combined[..., None] + concept_neg * (
            1 - combined[..., None]
        )

        x = self.concept_reasoner(embedding, combined)

        entropy_loss = F.mse_loss(x, self.diff_argmax(x), reduction="mean")

        if return_explanation:
            explanation = self.concept_reasoner.explain(
                embedding, combined, mode="global", concept_names=concept_names
            )
        else:
            explanation = None

        return x, entropy_loss, explanation


class GlobalConceptReasonerModule(torch.nn.Module):
    def __init__(self, n_concepts, n_classes, config: Config | BronzeConfig):
        super(GlobalConceptReasonerModule, self).__init__()
        self.n_classes = n_classes
        self.concept_reasoner = GlobalConceptReasoningLayer(
            n_concepts, n_classes, temperature=config.concept_temperature
        )

        self.diff_argmax = DifferentiableArgmax(config)

    def forward(self, combined, return_explanation=False, concept_names=None):
        x = self.concept_reasoner(combined)

        if return_explanation:
            explanation = self.concept_reasoner.explain(
                combined, mode="global", concept_names=concept_names
            )
        else:
            explanation = None

        entropy_loss = F.mse_loss(x, self.diff_argmax(x), reduction="mean")

        return x, entropy_loss, explanation


class MemoryBasedReasonerModule(torch.nn.Module):
    def __init__(self, n_concepts, n_classes, config: BronzeConfig):
        super(MemoryBasedReasonerModule, self).__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_rules = config.concept_memory_disjunctions
        self.emb_size = config.concept_embedding_size

        self.rule_book = nn.Embedding(n_classes * self.n_rules, self.emb_size)

        self.rule_decoder = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.LeakyReLU(),
            nn.Linear(self.emb_size, 3 * n_concepts),
        )

        """self.rule_decoder_highend = nn.Embedding(n_classes * n_rules * n_concepts, 3)"""

        self.rule_selector = nn.Sequential(
            nn.Linear(self.n_concepts, self.emb_size),
            nn.LeakyReLU(),
            nn.Linear(self.emb_size, n_classes * self.n_rules),
        )

        self.diff_argmax = DifferentiableArgmax(config)

    def decode_rules(self):
        rule_embs = self.rule_book.weight.view(
            self.n_classes, self.n_rules, self.emb_size
        )
        rules_decoded = self.rule_decoder(rule_embs).view(
            self.n_classes, self.n_rules, self.n_concepts, 3
        )
        # rules_decoded = self.rule_decoder_highend.weight.view(self.n_classes, self.n_rules, self.n_concepts, 3)
        rules_decoded = F.softmax(rules_decoded, dim=-1)
        if not self.training:
            # argmax to get the most likely rule
            rules_decoded = F.one_hot(
                torch.argmax(rules_decoded, dim=-1), num_classes=3
            ).float()
        return rules_decoded

    def forward(self, x, return_explanation=False, concept_names=None):

        rules_decoded = self.decode_rules()
        rule_scores = self.rule_selector(x).view(-1, self.n_classes, self.n_rules)
        rule_scores = F.softmax(rule_scores, dim=-1)  # (batch_dim, n_classes, n_rules)
        if not self.training:
            # argmax to get the most likely rule
            rule_scores = F.one_hot(
                torch.argmax(rule_scores, dim=-1), num_classes=self.n_rules
            ).float()

        agg_rules = (
            rules_decoded[None, ...] * rule_scores[..., None, None]
        )  # # (batch_dim, n_classes, n_rules, n_concepts, 3)
        agg_rules = agg_rules.sum(dim=-3)
        pos_rules = agg_rules[..., 0]
        neg_rules = agg_rules[..., 1]
        irr_rules = agg_rules[..., 2]
        x = x[..., None, :]
        # batch_dim, n_classes, n_concepts
        preds = (pos_rules * x + neg_rules * (1 - x) + irr_rules).prod(dim=-1)
        preds = preds.clamp(0.001, 0.999)
        c_rec = 0.5 * irr_rules + pos_rules
        c_rec = c_rec.clamp(0.001, 0.999)
        # c_rec_w = (1 - irr_rules)
        # aux_loss = (F.binary_cross_entropy(pos_rules, x.repeat(1, pos_rules.shape[1], 1),reduction="none") * c_rec_w).mean(dim=-1)
        aux_loss = F.binary_cross_entropy(
            c_rec, x.repeat(1, c_rec.shape[1], 1), reduction="none"
        ).mean(dim=-1)

        entropy_loss = F.mse_loss(preds, self.diff_argmax(preds), reduction="mean")
        aux_loss = (aux_loss * preds).mean() + entropy_loss

        explanations = None
        assert (
            not return_explanation or not self.training
        ), "Explanation can only be returned in eval mode"
        if return_explanation:
            if concept_names is None:
                concept_names = [f"c_{i}" for i in range(self.n_concepts)]
            rule_counts = (
                rule_scores.round().to(torch.long)
                * preds[..., None].round().to(torch.long)
            ).sum(
                dim=0
            )  # (n_classes, n_rules)
            from collections import defaultdict

            rule_strings = defaultdict(list)
            explanations = {}
            for i in range(self.n_classes):
                class_name = f"y_{i}"
                for j in range(self.n_rules):
                    for c in range(self.n_concepts):
                        is_pos = rules_decoded[i, j, c, 0]
                        is_neg = rules_decoded[i, j, c, 1]
                        is_irr = rules_decoded[i, j, c, 2]
                        if is_pos > 0.5:
                            rule_strings[(i, j)].append(concept_names[c])
                        elif is_neg > 0.5:
                            rule_strings[(i, j)].append(f"~{concept_names[c]}")
                explanations[class_name] = []
                for j in range(self.n_rules):
                    if rule_counts[i, j] > 0:
                        rule_str = " & ".join(rule_strings[(i, j)])
                        explanations[class_name].append(
                            {"rule": rule_str, "count": int(rule_counts[i, j].item()), "index": j}
                        )

        return preds, aux_loss, explanations
