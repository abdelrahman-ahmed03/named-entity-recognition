from __future__ import annotations

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """Linear-chain Conditional Random Field for sequence labeling."""

    def __init__(self, number_of_tags: int) -> None:
        super().__init__()

        if number_of_tags <= 0:
            raise ValueError(
                "number_of_tags must be greater than zero."
            )

        self.number_of_tags = number_of_tags

        self.start_transitions = nn.Parameter(
            torch.empty(number_of_tags)
        )

        self.end_transitions = nn.Parameter(
            torch.empty(number_of_tags)
        )

        self.transitions = nn.Parameter(
            torch.empty(
                number_of_tags,
                number_of_tags,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.start_transitions,
            mean=0.0,
            std=0.1,
        )

        nn.init.normal_(
            self.end_transitions,
            mean=0.0,
            std=0.1,
        )

        nn.init.normal_(
            self.transitions,
            mean=0.0,
            std=0.1,
        )

    def _validate_inputs(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                "emissions must have shape "
                "(batch_size, sequence_length, number_of_tags)."
            )

        if emissions.size(-1) != self.number_of_tags:
            raise ValueError(
                "The final emissions dimension must match "
                "number_of_tags."
            )

        if mask.dim() != 2:
            raise ValueError(
                "mask must have shape "
                "(batch_size, sequence_length)."
            )

        if emissions.shape[:2] != mask.shape:
            raise ValueError(
                "The first two dimensions of emissions "
                "must match the mask shape."
            )

        if tags is not None and tags.shape != mask.shape:
            raise ValueError(
                "tags and mask must have the same shape."
            )

        if not mask[:, 0].all():
            raise ValueError(
                "The first token of every sequence "
                "must be active in the mask."
            )

    def _calculate_gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = emissions.shape

        batch_indices = torch.arange(
            batch_size,
            device=emissions.device,
        )

        score = self.start_transitions[tags[:, 0]]

        score = score + emissions[
            batch_indices,
            0,
            tags[:, 0],
        ]

        for time_step in range(1, sequence_length):
            active = mask[:, time_step]

            transition_score = self.transitions[
                tags[:, time_step - 1],
                tags[:, time_step],
            ]

            emission_score = emissions[
                batch_indices,
                time_step,
                tags[:, time_step],
            ]

            score = score + (
                transition_score + emission_score
            ) * active

        last_indices = mask.long().sum(dim=1) - 1

        last_tags = tags.gather(
            dim=1,
            index=last_indices.unsqueeze(1),
        ).squeeze(1)

        score = score + self.end_transitions[last_tags]

        return score

    def _calculate_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        score = (
            self.start_transitions
            + emissions[:, 0]
        )

        for time_step in range(
            1,
            emissions.size(1),
        ):
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[
                    :, time_step
                ].unsqueeze(1)
            )

            next_score = torch.logsumexp(
                next_score,
                dim=1,
            )

            score = torch.where(
                mask[:, time_step].unsqueeze(1),
                next_score,
                score,
            )

        score = score + self.end_transitions

        return torch.logsumexp(
            score,
            dim=1,
        )

    def negative_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.bool()

        self._validate_inputs(
            emissions=emissions,
            tags=tags,
            mask=mask,
        )

        gold_score = self._calculate_gold_score(
            emissions=emissions,
            tags=tags,
            mask=mask,
        )

        normalizer = self._calculate_normalizer(
            emissions=emissions,
            mask=mask,
        )

        return (
            normalizer - gold_score
        ).mean()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        mask = mask.bool()

        self._validate_inputs(
            emissions=emissions,
            tags=None,
            mask=mask,
        )

        score = (
            self.start_transitions
            + emissions[:, 0]
        )

        history: list[torch.Tensor] = []

        for time_step in range(
            1,
            emissions.size(1),
        ):
            next_score = (
                score.unsqueeze(2)
                + self.transitions.unsqueeze(0)
            )

            best_score, best_previous_tag = (
                next_score.max(dim=1)
            )

            best_score = (
                best_score
                + emissions[:, time_step]
            )

            score = torch.where(
                mask[:, time_step].unsqueeze(1),
                best_score,
                score,
            )

            history.append(best_previous_tag)

        score = score + self.end_transitions

        best_last_tags = score.argmax(dim=1)

        sequence_lengths = mask.long().sum(
            dim=1
        )

        decoded_sequences: list[list[int]] = []

        for batch_index in range(
            emissions.size(0)
        ):
            length = int(
                sequence_lengths[batch_index].item()
            )

            best_tag = int(
                best_last_tags[batch_index].item()
            )

            best_path = [best_tag]

            relevant_history = history[
                :max(length - 1, 0)
            ]

            for previous_tags in reversed(
                relevant_history
            ):
                best_tag = int(
                    previous_tags[
                        batch_index,
                        best_tag,
                    ].item()
                )

                best_path.append(best_tag)

            best_path.reverse()
            decoded_sequences.append(best_path)

        return decoded_sequences