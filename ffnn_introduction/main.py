"""Main example script."""

import datetime

import jax
import jax.random as jrandom
import klax
from matplotlib import pyplot as plt
import time

import src.tmlsm.losses as tl
import src.tmlsm.data as td
import src.tmlsm.models as tm

now = datetime.datetime.now


def main():
    # Create random key for reproducible weight initialization, and
    # batch splits. The call to `time.time_ns()` may be replaced with
    # a constant seed if exactly reproductible results ought to be
    # produced.
    key = jrandom.PRNGKey(time.time_ns())
    keys = jrandom.split(key, 2)

    # Build model instance
    model = tm.build(key=keys[0])
    print(model)

    # Load data
    x, y, x_cal, y_cal = td.bathtub()

    # Calibrate model
    t1 = now()
    print(t1)

    model, history = klax.fit(
        model,
        (x_cal, y_cal),
        batch_size=32,
        steps=1_000,
        loss_fn=tl.MSE(),
        history=klax.HistoryCallback(log_every=1),
        key=keys[1],
    )

    t2 = now()
    print("it took", t2 - t1, "(sec) to calibrate the model")

    history.plot()

    # Evaluation
    # First the model need to be finalized to unwrap and apply all
    # wrappers and constraints (if present).
    model_ = klax.finalize(model)

    plt.figure(2)
    plt.scatter(x_cal[::10], y_cal[::10], c="green", label="calibration data")
    plt.plot(x, y, c="black", linestyle="--", label="bathtub function")
    plt.plot(x, jax.vmap(model_)(x), label="model", color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
