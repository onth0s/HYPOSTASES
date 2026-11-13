"""Background NNUE training daemon thread management."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from hypostases.plugins.domains.chess.nnue_net import NNUENet
from hypostases.plugins.domains.chess.nnue_training import train_nnue

if TYPE_CHECKING:
    import chess


def extract_nnue_weights(net: NNUENet) -> dict[str, np.ndarray]:
    """Extract weight matrices and bias vectors from an NNUENet instance."""
    return {
        "W_white": net.W_white.copy(),
        "W_black": net.W_black.copy(),
        "W_l1": net.W_l1.copy(),
        "b_l1": net.b_l1.copy(),
        "W_l2": net.W_l2.copy(),
        "b_l2": net.b_l2.copy(),
    }


class NNUETrainingDaemon:
    """Manages asynchronous background SGD training of NNUE weights."""

    def __init__(
        self,
        initial_net: NNUENet,
        epochs: int = 30,
        learning_rate: float = 0.003,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.committed_lock = threading.Lock()
        self.committed_weights: dict[str, np.ndarray] = extract_nnue_weights(initial_net)
        self.sgd_ev: threading.Event | None = None
        self.sgd_gen: int | None = None
        self.sgd_loss: float | None = None

    def get_committed_weights(self) -> dict[str, np.ndarray]:
        """Thread-safe snapshot copy of current committed weights."""
        with self.committed_lock:
            return {k: v.copy() for k, v in self.committed_weights.items()}

    def kick_background_sgd(
        self,
        gen_idx: int,
        train_positions: list[tuple[chess.Board, float]],
    ) -> None:
        """Starts NNUE SGD on a daemon thread; publishes the net only on completion."""
        if not train_positions:
            return

        with self.committed_lock:
            base = {k: v.copy() for k, v in self.committed_weights.items()}

        dataset = list(train_positions)
        self.sgd_gen = gen_idx
        self.sgd_loss = None
        ev = threading.Event()

        def _worker() -> None:
            net = NNUENet()
            for key, val in base.items():
                setattr(net, key, val.copy())
            loss = train_nnue(
                net,
                dataset,
                epochs=self.epochs,
                lr=self.learning_rate,
                verbose=False,
            )
            with self.committed_lock:
                self.committed_weights.clear()
                self.committed_weights.update(extract_nnue_weights(net))
            self.sgd_loss = float(loss)
            ev.set()

        threading.Thread(
            target=_worker,
            name=f"nnue-sgd-gen{gen_idx}",
            daemon=True,
        ).start()
        self.sgd_ev = ev

    def drain_sgd_result(self) -> tuple[int | None, float | None]:
        """Polls whether the background SGD run completed.

        Returns (gen_idx, loss) if completed, or (None, None) if in progress / none active.
        """
        if self.sgd_ev is not None and self.sgd_ev.is_set():
            gen = self.sgd_gen
            loss = self.sgd_loss
            self.sgd_ev = None
            self.sgd_gen = None
            self.sgd_loss = None
            return gen, loss
        return None, None
