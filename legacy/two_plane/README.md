# Legacy: 2-plane G(2,3) parameterization

Reference-only archive. Files here implement the older 2-plane
parameterization where each Gaussian's covariance was a pair $(p, q)$
of unit imaginary quaternions and the plane $E_{p,q}$ was built via
the Jacobian-paper Definition 2 shorthands $(c, d, s, r)$.

- `quaternion.py`: Hamilton-product / pure-imaginary / unit-vector
  primitives used by `grassmann.py`.
- `grassmann.py`: canonical frame $E_{p,q}$, line-to-(p,q) and
  (p,q)-to-line correspondences.
- `test_quaternion.py`, `test_grassmann.py`: unit tests for the above.

The v7 spec (current) uses the 3-plane $G(3,4)$ parameterization
implemented in `grassmann/gaussian.py`. Nothing on the live training
path references the modules archived here.
