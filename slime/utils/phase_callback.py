"""Driver-level phase callbacks for the Slime training loop.

Subclass ``PhaseCallback`` and pass your class via ``--phase-callback-path``
to receive notifications at GPU phase boundaries (init, generate, train,
weight_sync).  The driver instantiates your class once and calls
``on_phase_begin``/``on_phase_end`` around each phase.

Example usage::

    # my_callbacks.py
    from slime.utils.phase_callback import PhaseCallback

    class ProfilingCallback(PhaseCallback):
        def on_phase_begin(self, phase, role, context=None):
            print(f"[begin] {phase} ({role})")
        def on_phase_end(self, phase, role, context=None):
            print(f"[end]   {phase} ({role})")

    # launch command
    python train.py ... --phase-callback-path my_callbacks.ProfilingCallback
"""


class PhaseCallback:
    """Base class for phase-level callbacks.

    Override ``on_phase_begin`` and ``on_phase_end`` to act at phase boundaries.

    Parameters passed to callbacks:

    *phase* — one of:
      ``"init"``        — model loading + first weight sync (both pools)
      ``"generate"``    — rollout/inference (sampler GPUs)
      ``"train"``       — forward/backward/optimizer + save_model (trainer GPUs)
      ``"weight_sync"`` — cross-pool NCCL weight broadcast (both pools)
      ``"offload"``     — GPU→CPU memory migration (sampler GPUs)
      ``"onload"``      — CPU→GPU memory restore (sampler GPUs)
      ``"eval"``        — evaluation inference (sampler GPUs)
      ``"create"``      — actor recreation in release_train mode (trainer GPUs)

    *role* — which GPU pool is active: ``"trainer"``, ``"sampler"``,
    or ``"both"`` (e.g. weight sync touches trainer and sampler GPUs).

    *context* — optional dict with phase metadata (``rollout_id``, etc.).
    """

    def on_phase_begin(self, phase: str, role: str, context: dict | None = None) -> None:
        pass

    def on_phase_end(self, phase: str, role: str, context: dict | None = None) -> None:
        pass

    def close(self) -> None:
        """Release resources.  Called once after the training loop exits."""
        pass


def load_phase_callback(args):
    """Instantiate a PhaseCallback from ``--phase-callback-path``, or None."""
    path = getattr(args, "phase_callback_path", None)
    if not path:
        return None
    from slime.utils.misc import load_function

    cls = load_function(path)
    return cls()
