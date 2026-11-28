"""Multiscale representation controller: deciding what the exact solver is handed.

The engine underneath is a correct exact stochastic solver. Nothing here replaces it.
Everything here prices an approximation BEFORE it is made, on the one idea that an
assumption is a CUT: a species held at its mean stops transmitting fluctuations, so
everything downstream of it decouples, and cuts are what an elimination engine turns
into speed.

Two classes of approximation, and the distinction is load-bearing:
    perturbs a rate   -- bounded, analytic, roughly 7x worse in the tail than the mean
    deletes a pathway -- UNBOUNDED
"""
