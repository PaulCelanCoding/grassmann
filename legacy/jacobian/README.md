# Legacy: 2-plane analytic Jacobian

Reference-only archive. Files here implement and test the analytic
$\partial(p, q)/\partial(\text{image})$ Jacobian used by the older
2-plane $G(2, 3)$ parameterization.

The current code lives in `grassmann/` and targets the 3-plane
$G(3, 4)$ parameterization of the v7 math spec; its only projection
Jacobian is $J_\pi$, implemented in `grassmann/projection.py`.

Imports in this folder still point at `grassmann.jacobian`, which no
longer exists on the live path — they are kept verbatim from the
2-plane era so the math (and the tests against autograd) remain
auditable. No attempt is made to make these files pytest-collectable
from `legacy/`.
