import abc
from collections import Counter

import torch

from bronze_age.config import Config


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
    


class GlobalConceptReasoningLayer(torch.nn.Module):
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
        self.filter_nn = torch.nn.Embedding(n_concepts, n_classes)
        self.sign_nn = torch.nn.Embedding(n_concepts, n_classes)
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

    def explain(
        self, c, mode, concept_names=None, class_names=None, filter_attn=None
    ):
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
                        sign_attn = sign_attn_mask[
                            0, concept_idx, target_class
                        ]
                        filter_attn = filter_attn_mask[
                            0, concept_idx, target_class
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


def generate_names(n_concepts, in_channels, bounding_parameter):
    names = [f"s_{i}" for i in range(n_concepts)] + [
        f"s_{i}_count>{j}"
        for i in range(in_channels)
        for j in range(bounding_parameter)
    ]
    return names


class ConceptReasonerModule(torch.nn.Module):
    def __init__(self, n_concepts, n_classes, emb_size, config: Config):
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

        if return_explanation:
            explanation = self.concept_reasoner.explain(
                embedding, combined, mode="global", concept_names=concept_names
            )
            return x, explanation

        return x


class GlobalConceptReasonerModule(torch.nn.Module):
    def __init__(self, n_concepts, n_classes, config: Config):
        super(GlobalConceptReasonerModule, self).__init__()
        self.n_classes = n_classes
        self.concept_reasoner = GlobalConceptReasoningLayer(
            n_concepts, n_classes, temperature=config.concept_temperature
        )

    def forward(self, combined, return_explanation=False, concept_names=None):
        x = self.concept_reasoner(combined)

        if return_explanation:
            explanation = self.concept_reasoner.explain(
                combined, mode="global", concept_names=concept_names
            )
            return x, explanation

        return x