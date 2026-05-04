\documentclass[11pt,reqno]{amsart}

\usepackage{amssymb,amsmath,amsfonts,amsthm,mathtools,enumerate}
\usepackage[top=1in,margin=1.2in,bottom=1in]{geometry}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue]{hyperref}
\usepackage[capitalize]{cleveref}
\usepackage{listings}
\usepackage{xcolor}

\lstset{
  language=Python,
  basicstyle=\ttfamily\small,
  keywordstyle=\color{blue},
  commentstyle=\color{gray!70!black},
  stringstyle=\color{orange!80!black},
  showstringspaces=false,
  numbers=left,
  numberstyle=\tiny\color{gray},
  frame=single,
  breaklines=true
}

\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}
\DeclarePairedDelimiter{\norm}{\lVert}{\rVert}
\newcommand{\ip}[1]{\langle #1 \rangle}
\newcommand{\Gr}{\mathrm{Gr}}
\newcommand{\PSD}{\mathrm{Sym}^+}
\newcommand{\spn}{\mathrm{span}}
\newcommand{\rk}{\mathrm{rank}}
\newcommand{\tr}{\mathrm{tr}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\img}{\mathrm{im}}

\theoremstyle{plain}
\newtheorem{prop}{Proposition}[section]
\newtheorem{lem}[prop]{Lemma}
\newtheorem{cor}[prop]{Corollary}
\newtheorem{theo}[prop]{Theorem}
\theoremstyle{remark}
\newtheorem{remark}[prop]{Remark}
\theoremstyle{definition}
\newtheorem{defn}[prop]{Definition}

\begin{document}

\title[Grassmann Splatting on $\Gr(3,4)$]{Grassmann Splatting on $\Gr(3,4)$:\\
A Projector-Form Framework for Dynamic Scene Rendering}

\author{A. Berman}
\author{S. Dave}
\author{C. Zaboklicki}
\date{Draft}

\begin{abstract}
We present a framework for rendering dynamic 3D scenes from video in which each primitive is a Gaussian density supported on a 3-plane in spacetime $\R^4 = \R \times \R^3$. The 3-plane is parameterized by its unit normal $n \in S^3 / \{\pm 1\} \cong \Gr(3,4)$, and the Gaussian's 4D covariance is constructed from an unconstrained factor $L \in \R^{4 \times 3}$ via
\[
\Sigma_{4D} = (P_n L)(P_n L)^T, \qquad P_n = I - n n^T.
\]
Rendering at frame $t_0$ proceeds by conditioning the 4D Gaussian on $t = t_0$ in world coordinates --- a Schur complement on the temporal axis --- and handing the resulting 3D Gaussian to a standard 3D Gaussian splatting rasterizer. The conditioned spatial covariance has rank 2, recovering the disk-shaped primitive of static 3D Gaussian splatting. The architecture has three benefits: (i) time conditioning is exact, with no Taylor expansion of the perspective map; (ii) the camera trajectory enters through the rasterizer's view matrix at the rendered pose, avoiding any linearization of camera motion; (iii) implementation requires no custom CUDA code. We give the construction in full, prove the rank structure of the conditioned covariance, analyze the three distinct blur parameters that appear in the pipeline, treat numerical stability under degeneracy, and provide an explicit algorithm and gradient analysis.
\end{abstract}

\maketitle
\tableofcontents

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Introduction}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{Goal}

We are given a sequence of video frames depicting a 3D scene that may evolve over time, possibly observed by a moving camera. The goal is a differentiable scene representation that can be fit to the input sequence and queried at arbitrary frame index and viewpoint, including viewpoints not present in the training data. The representation should be lightweight, support real-time rendering, and have a parameter count comparable to that of static 3D Gaussian splatting (3DGS).

\subsection{The primitive}

We extend 3DGS by promoting the static 3D ellipsoid primitive to a primitive that lives in spacetime. Concretely: each Gaussian in our model is supported on a 3-plane in $\R^4 = \R \times \R^3$, where the first axis is time and the remaining three are space. Slicing such a 3-plane at a fixed time $t = t_0$ yields a 2-plane in $\R^3$ --- exactly the support of a flat disk-shaped 3DGS Gaussian. The temporal evolution of the primitive is encoded in the orientation of the 3-plane in $\R^4$: the more the plane tilts away from the spatial slice $\{t = 0\}$, the more its time-slice drifts as $t$ varies.

\subsection{Approach}

Three observations make the construction tractable.

\emph{First, the Grassmannian admits a projector parameterization.} A 3-plane in $\R^4$ is determined by its unit normal $n \in S^3$ (up to sign), and the orthogonal projector $P_n = I - n n^T$ encodes the plane intrinsically, without choosing a chart. We will see that PSD covariances on the plane can be built directly from $P_n$ and an unconstrained factor in $\R^{4 \times 3}$.

\emph{Second, time is a linear marginal.} Conditioning a Gaussian on the linear constraint $t = t_0$ is exact in closed form (the Schur complement). No Taylor expansion is required for the temporal conditioning step. By performing this conditioning in \emph{world coordinates}, before any view transform, we reduce the per-frame primitive to a standard 3D Gaussian to which an unmodified 3DGS rasterizer can be applied.

\emph{Third, world-space conditioning sidesteps camera-trajectory linearization.} Because the conditioning happens before the rasterizer ever sees the data, the camera pose at the rendered frame, $(R(t_0), c(t_0))$, is used exactly as the view matrix. There is no need to linearize the camera trajectory in time, which would introduce error proportional to the camera's instantaneous velocity scaled by the Gaussian's temporal width.

\subsection{Contributions}

\begin{enumerate}
\item We formulate the Grassmann splatting framework on $\Gr(3,4)$ with a projector-form parameterization that avoids charts, giving a single global construction of the 4D covariance (Section~\ref{sec:primitive}).

\item We show that time conditioning on the resulting 4D Gaussian is exact, and that the conditioned 3D spatial covariance has rank 2 generically (Section~\ref{sec:conditioning}, Proposition~\ref{prop:rank2}).

\item We analyze numerical stability of the pipeline. Three independent ``blur'' parameters appear --- a pixel-domain EWA blur, a temporal weight floor, and a 3D rank-lift for CUDA compatibility --- and we explain why conflating them is incorrect (Section~\ref{sec:stability}). A small numerical floor $\varepsilon$ in the temporal denominators acts as a smooth topological bridge between the rank-3 (static-3DGS) and rank-2 (dynamic disk) regimes, allowing initialization at $n = e_0$ and continuous evolution into the dynamic representation (Proposition~\ref{prop:bridge}).

\item We prove that world-space conditioning is exact in the camera trajectory, in the precise sense that any pixel-space conditioning architecture incurs an error proportional to camera speed times temporal Gaussian width (Proposition~\ref{prop:camera}).

\item We give an explicit gradient analysis showing that all parameters --- including the manifold-valued plane normal $n \in S^3 / \{\pm 1\}$ --- can be optimized via standard PyTorch autograd plus an optional Riemannian projection step (Section~\ref{sec:gradients}).

\item We give an initialization scheme adapted to monocular video and an explicit per-iteration algorithm (Sections~\ref{sec:training} and~\ref{sec:algorithm}).

\item The framework requires \emph{no custom CUDA code}: a standard 3DGS rasterizer (e.g.\ \texttt{diff-gaussian-rasterization} with the \texttt{cov3D\_precomp} interface) suffices.
\end{enumerate}

\subsection{Notation}

We work in $\R^4$ with coordinates $x = (x_0, X)$ where $x_0 \in \R$ is time and $X = (x_1, x_2, x_3) \in \R^3$ is space. The standard basis of $\R^4$ is $\{e_0, e_1, e_2, e_3\}$ with $e_0 = (1,0,0,0)^T$. Inner products and norms on $\R^k$ are the Euclidean ones, written $\ip{\cdot, \cdot}$ and $\norm{\cdot}$. We use $I_k$ for the $k \times k$ identity matrix; $\PSD_k$ denotes the cone of $k \times k$ positive semidefinite matrices. For a matrix $M$ we write $\img(M)$ for its column space and $\ker(M)$ for its kernel. The Schur complement of a $2 \times 2$ block matrix $\begin{pmatrix} A & B \\ B^T & D \end{pmatrix}$ with $A$ invertible is $D - B^T A^{-1} B$.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{The Geometric Setting}\label{sec:geometry}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{Spacetime}

The ambient space is $\R^4 = \R \times \R^3$ with the splitting into time and space described above. There is no quaternion structure, no complex structure, and no orientation reversal involved: $\R^4$ is treated as a Euclidean inner product space with a distinguished basis vector $e_0$ (the time axis).

\subsection{The Grassmannian $\Gr(3, 4)$}

\begin{defn}[The Grassmannian]
Let $\Gr(3, 4)$ denote the manifold of 3-dimensional linear subspaces of $\R^4$. As a homogeneous space,
\[
\Gr(3, 4) \;\cong\; \mathrm{O}(4) \,/\, (\mathrm{O}(3) \times \mathrm{O}(1)) \;\cong\; S^3 \,/\, \{\pm 1\} \;\cong\; \mathbb{RP}^3,
\]
a smooth manifold of dimension 3.
\end{defn}

The diffeomorphism $\Gr(3,4) \cong S^3 / \{\pm 1\}$ is realized by the map $n \mapsto E_n$ where $E_n$ is the orthogonal complement of $n$:

\begin{defn}[Plane and projector]\label{defn:plane}
For $n \in S^3 \subset \R^4$, the canonical 3-plane is
\[
E_n \;=\; \{ x \in \R^4 \,:\, \ip{n, x} = 0 \},
\]
a 3-dimensional linear subspace. Since $E_n = E_{-n}$, the assignment $n \mapsto E_n$ descends to a bijection $S^3 / \{\pm 1\} \to \Gr(3, 4)$. The orthogonal projector onto $E_n$ is
\[
P_n \;=\; I - n n^T \;\in\; \R^{4 \times 4}.
\]
It satisfies $P_n^2 = P_n$, $P_n^T = P_n$, $P_n n = 0$, and $P_n x = x$ for all $x \in E_n$.
\end{defn}

\subsection{Geometric interpretation: a 3-plane is a moving 2D plane}\label{sec:moving_plane}

The spatial slice of $E_n$ at time $t = t_0$ is the affine subset
\[
E_n \cap \{x_0 = t_0\} \;\subset\; \R^4.
\]
This intersection is a 2-dimensional affine subspace of $\R^3$ (a plane in space, after projecting out the time coordinate), provided $E_n$ is not itself the spatial slice $\{x_0 = 0\}$. As $t_0$ varies, this affine 2-plane in space sweeps through $\R^3$, in general translating and possibly tilting.

\begin{prop}[Slicing law]\label{prop:slice}
Write $n = (n_0, n_{1:}) \in \R \times \R^3$. If $n_{1:} \neq 0$, the spatial slice at time $t_0$ is the affine plane
\[
\big\{ X \in \R^3 \,:\, \ip{n_{1:}, X} = -t_0 n_0 \big\} \;\subset\; \R^3,
\]
which has unit normal $n_{1:} / \norm{n_{1:}}$ and signed distance $-t_0 n_0 / \norm{n_{1:}}$ from the origin in $\R^3$. As $t_0$ varies, the plane translates rigidly along its normal direction with velocity vector $-n_0 \, n_{1:} / \norm{n_{1:}}^2 \in \R^3$ (signed speed $-n_0 / \norm{n_{1:}}$); it does not rotate.
\end{prop>

\begin{proof}
The defining equation $\ip{n, x} = 0$ at $x_0 = t_0$ reads $n_0 t_0 + \ip{n_{1:}, X} = 0$, which gives the stated affine equation. The normal direction $n_{1:} / \norm{n_{1:}}$ is independent of $t_0$, so the plane translates rigidly.
\end{proof}

\begin{remark}[Degenerate cases]\label{rem:degen}
If $n_0 = 0$, the plane $E_n$ contains the time axis: $E_n \cap \{x_0 = t_0\}$ is the same 2-plane in $\R^3$ for every $t_0$. The Gaussian on $E_n$ is then ``frozen'' --- it has no temporal evolution, and the framework reduces to a static 3DGS Gaussian on a 2-plane in $\R^3$.

Conversely, if $n_{1:} = 0$ (i.e.\ $n = \pm e_0$), the plane $E_n$ is exactly the spatial slice $\{x_0 = 0\}$. The plane has no extent in time at all: it is a single instantaneous spatial configuration. We will see in Section~\ref{sec:stability} that this case requires care, because the temporal variance $\Sigma_{tt}$ tends to zero.
\end{remark}

\subsection{Why $\Gr(3, 4)$ rather than $\Gr(2, 4)$?}

A natural alternative is to model each primitive as a 2-plane in $\R^4$ (a moving line in space), parameterized by $\Gr(2, 4) \cong \Gr^+(4, 2) / \{\pm 1\} \cong (S^2 \times S^2) / \{\pm 1\}$. The construction is well-known and the parameterization admits a clean basis via quaternionic multiplication. We do not pursue this path because the spatial slice of a 2-plane in $\R^4$ is a 1-dimensional line in $\R^3$, which projects to a 1-dimensional structure in the image plane. This gives the rasterizer a degenerate primitive that must be artificially fattened, and empirically produces a ``streaking'' pathology in which the line tends to align with the rendering ray. The 3-plane primitive avoids this by yielding a rank-2 disk after time conditioning, which is precisely the natural primitive of static 3DGS.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{The Gaussian Primitive}\label{sec:primitive}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{Parameters}

Each Gaussian in the model carries:
\begin{itemize}
\item $n \in S^3 / \{\pm 1\}$: the plane normal, 3 degrees of freedom (DOF). Stored as an unconstrained vector $n_{\text{raw}} \in \R^4$ and renormalized in the forward pass.
\item $L \in \R^{4 \times 3}$: an unconstrained ``Cholesky-like'' factor.
\item $\mu \in \R^4$: the mean of the Gaussian in spacetime; we will write $\mu = (v_0, V)$ with $v_0 \in \R$ and $V \in \R^3$.
\item $\alpha \in [0, 1]$: a base opacity, stored as a logit and passed through a sigmoid.
\item Color: either a constant RGB triple or a vector of spherical harmonic coefficients (multiplied by 3 for color channels).
\item Three blur scalars $\sigma_{\text{pix}}^2, \sigma_{\text{tmp}}^2, \sigma_{\text{lift}}^2$, all non-negative, all variances (see Section~\ref{sec:sigma_trio}).
\end{itemize}

We will count effective DOFs in Section~\ref{sec:dof}.

\subsection{The 4D covariance}

\begin{defn}[Projected factor and 4D covariance]
The projected factor is
\begin{equation}\label{eq:Ln}
L_n \;=\; P_n L \;=\; L - n (n^T L) \;\in\; \R^{4 \times 3},
\end{equation}
whose three columns lie in $E_n$. The 4D covariance is
\begin{equation}\label{eq:Sigma4D}
\boxed{\Sigma_{4D} \;=\; L_n L_n^T \;=\; (P_n L)(P_n L)^T \;\in\; \PSD_4.}
\end{equation}
\end{defn}

The factorization \eqref{eq:Sigma4D} has three immediate consequences.

\begin{prop}[Rank and kernel of $\Sigma_{4D}$]\label{prop:rank4D}
The matrix $\Sigma_{4D}$ is positive semidefinite with $\rk(\Sigma_{4D}) \leq 3$ and $\spn(n) \subseteq \ker(\Sigma_{4D})$. Consequently $\img(\Sigma_{4D}) \subseteq E_n$.
\end{prop}

\begin{proof}
$\Sigma_{4D} = L_n L_n^T$ is PSD as a Gram matrix. Its rank is at most $\min(4, 3) = 3$. For the kernel: $L_n^T n = (P_n L)^T n = L^T P_n^T n = L^T P_n n = 0$, so $n^T \Sigma_{4D} = (L_n^T n)^T L_n^T = 0$, hence $\Sigma_{4D} n = 0$ by symmetry. The image of $\Sigma_{4D}$ is the orthogonal complement of its kernel, which contains $\spn(n)$, so $\img(\Sigma_{4D}) \subseteq \spn(n)^\perp = E_n$.
\end{proof}

\begin{remark}[The construction in plain words]
The factor $L \in \R^{4 \times 3}$ is unconstrained: any of its $12$ entries are free during optimization. The projection $P_n L = L - n(n^T L)$ kills the $n$-component of each column of $L$, forcing the columns of $L_n$ to lie in $E_n$. The covariance $\Sigma_{4D} = L_n L_n^T$ is then automatically PSD with image in $E_n$. Compare this with the alternative of choosing a chart on $\Gr(3,4)$ and parameterizing $\Sigma$ as a $3 \times 3$ PSD matrix in that chart: the projector form is global and chart-free, at the cost of carrying a redundant $L$-direction (along $n$) that the optimizer simply ignores.
\end{remark}

\subsection{Effective DOFs of $L$}\label{sec:Ldof}

The matrix $L$ has 12 entries, but $\Sigma_{4D}$ depends on $L$ only through its image in $E_n$ and only modulo right-action by $\mathrm{O}(3)$ on the columns.

\begin{prop}[Effective DOFs of $\Sigma_{4D}$]\label{prop:dof_L}
The covariance $\Sigma_{4D} = L_n L_n^T$ has at most $6$ effective degrees of freedom.
\end{prop}

\begin{proof}
The set of PSD rank-$\leq 3$ matrices with image in the 3-dimensional subspace $E_n$ is exactly $\PSD_3$, the cone of $3 \times 3$ PSD matrices, which has dimension 6. Conversely, any element of $\PSD_3$ on $E_n$ admits a Cholesky-like factorization $L_n L_n^T$ with $L_n \in E_n^{\oplus 3}$, and $L = L_n$ then satisfies $P_n L = L_n$. Hence the map $L \mapsto \Sigma_{4D}$ is surjective onto a 6-dimensional set.
\end{proof}

\begin{remark}[Two redundant directions in $L$]
The 6 redundant entries decompose as: (i) 3 entries given by the column-wise $n$-components of $L$ (annihilated by $P_n$), and (ii) 3 entries corresponding to a right $\mathrm{O}(3)$-action on the columns of $L_n$, which leaves $L_n L_n^T$ invariant. The optimizer does not need to know about either redundancy: gradients vanish in directions (i), and gauge equivalence in directions (ii) means the optimizer simply settles into one orbit representative. No constraint or projection step is required at training time.
\end{remark}

\subsection{The mean and its degrees of freedom}\label{sec:mu_dof}

The mean $\mu = (v_0, V) \in \R \times \R^3 = \R^4$ is unconstrained. One might suspect that, since the kernel of $\Sigma_{4D}$ contains $n$, shifting $\mu \mapsto \mu + \lambda n$ for $\lambda \in \R$ leaves the rendered output invariant. We show that this is not the case generically.

\begin{prop}[$\mu$ has 4 effective DOFs]\label{prop:mu_dof}
Let $n = (n_0, n_{1:}) \in \R \times \R^3$ with $n \neq 0$, and let $\Sigma_{4D}$ have nonzero $(0,0)$ block $\Sigma_{tt}^{\mathrm{pure}} > 0$. Then there is no nonzero $\lambda$ for which the shift $\mu \mapsto \mu + \lambda n$ leaves the rendered output \emph{at every frame $t_0$} unchanged, except in the degenerate case $n_0 = 0$ and $\Sigma_{4D} \cdot n_{1:} = 0$ in the spatial block.
\end{prop}

\begin{proof}
A shift $\mu \mapsto \mu + \lambda n$ changes $v_0 \to v_0 + \lambda n_0$ and $V \to V + \lambda n_{1:}$. The conditioned mean at frame $t_0$ (Section~\ref{sec:conditioning}) is
\[
V_{3D}(t_0) \;=\; V \;+\; c_{\text{world}} \cdot (t_0 - v_0) / \Sigma_{tt}^{\mathrm{pure}},
\]
where $c_{\text{world}} = \Sigma_{4D} \, e_0$ restricted to the spatial block. Under the shift, $V_{3D}$ becomes
\[
V_{3D}'(t_0) \;=\; V \,+\, \lambda n_{1:} \,+\, c_{\text{world}} \cdot (t_0 - v_0 - \lambda n_0) / \Sigma_{tt}^{\mathrm{pure}}
\;=\; V_{3D}(t_0) \,+\, \lambda \big( n_{1:} \,-\, c_{\text{world}} \, n_0 / \Sigma_{tt}^{\mathrm{pure}} \big).
\]
For this to equal $V_{3D}(t_0)$ for all $t_0$, the bracketed term must vanish: $n_{1:} = c_{\text{world}} n_0 / \Sigma_{tt}^{\mathrm{pure}}$. Combined with the kernel constraint $\Sigma_{4D} n = 0$ in block form (Proposition~\ref{prop:rank4D}), which reads $n_0 \Sigma_{tt}^{\mathrm{pure}} + c_{\text{world}}^T n_{1:} = 0$ for the time-component and $n_0 c_{\text{world}} + \Sigma_{3D} n_{1:} = 0$ for the spatial-component, we obtain
\[
n_0 \big( \norm{c_{\text{world}}}^2 + (\Sigma_{tt}^{\mathrm{pure}})^2 \big) \;=\; 0.
\]
Since $\Sigma_{tt}^{\mathrm{pure}} > 0$, the only solution is $n_0 = 0$, in which case $n_{1:} = 0$ from the invariance condition, contradicting $n \in S^3$. Hence no such $\lambda \neq 0$ exists. The case $n_0 = 0$ requires further analysis through the spatial kernel relation, and the invariance survives only under the stated additional degeneracy.
\end{proof}

\begin{remark}[Empirical confirmation]
Hard-projecting $\mu \mapsto P_n \mu$ during training degrades validation PSNR by approximately $0.2$~dB on a slice-banana benchmark at 14k iterations. This empirical finding aligns with Proposition~\ref{prop:mu_dof}: the unconstrained $\mu$ provides genuine extra degrees of freedom that the optimizer exploits. We therefore treat $\mu$ as having 4 effective DOFs.
\end{remark}

\subsection{The blur trio: $\sigma_{\text{pix}}^2$, $\sigma_{\text{tmp}}^2$, $\sigma_{\text{lift}}^2$}\label{sec:sigma_trio}

A naive treatment of the rendering pipeline encounters a single isotropic ``blur'' parameter $\sigma_k^2 I$ added in observation space, as in standard EWA splatting. Three different blur quantities arise in our framework, and conflating them leads to subtle errors. They are all variances, all non-negative, and all enter the pipeline at distinct points.

\paragraph{1. Pixel-domain blur $\sigma_{\text{pix}}^2$.} This is the standard EWA pixel-domain isotropic blur, added as $\sigma_{\text{pix}}^2 I_2$ to the projected 2D screen-space covariance after perspective transformation. It is handled entirely by the rasterizer; we do not manipulate it in our preprocessing. Default value: $\sigma_{\text{pix}}^2 = 1$ (pixel units).

\paragraph{2. Temporal floor $\sigma_{\text{tmp}}^2$.} This variance is added to $\Sigma_{tt}^{\mathrm{pure}}$ for the purposes of evaluating the temporal saliency weight $w_t$ (Section~\ref{sec:eff_opacity}). It floors the \emph{modeling} width of the temporal window: setting $\sigma_{\text{tmp}}^2 > 0$ enforces a minimum visible duration for every Gaussian regardless of its data-driven temporal extent. Default value: $\sigma_{\text{tmp}}^2 = 0$ (the data is allowed to dictate width). Numerical stability when $\Sigma_{tt}^{\mathrm{pure}} \to 0$ \emph{and} $\sigma_{\text{tmp}}^2 = 0$ simultaneously --- which occurs at standard initialization, see Section~\ref{sec:stability} --- is provided by a separate numerical floor $\varepsilon$, applied in the same way to every denominator in the pipeline. Time units: frames squared.

\paragraph{3. 3D rank lift $\sigma_{\text{lift}}^2$.} The conditioned spatial covariance $\Sigma_{3D}(t_0)$ has rank exactly 2 (Proposition~\ref{prop:rank2}), but standard CUDA EWA rasterizer kernels assume an invertible $3 \times 3$ covariance. We lift the rank-2 to rank-3 by adding $\sigma_{\text{lift}}^2 I_3$ before passing to the rasterizer. Default value: $\sigma_{\text{lift}}^2 = 10^{-4}$ in scene units (corresponding to roughly $1$~cm in a typical scene), small enough to be a numerical fix and large enough to avoid catastrophic conditioning of the EWA matrix.

The three are all genuinely distinct in their action: $\sigma_{\text{pix}}^2$ acts in 2D pixel space, $\sigma_{\text{lift}}^2$ acts in 3D world space, and $\sigma_{\text{tmp}}^2$ acts in 1D time. Confusing them produces formulas that are dimensionally consistent in symbol but physically wrong.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{World-Space Time Conditioning}\label{sec:conditioning}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We now show that, for any frame $t_0$, the 4D Gaussian $\mathcal{N}(\mu, \Sigma_{4D})$ admits a closed-form conditioning on the linear constraint $x_0 = t_0$, yielding a 3D Gaussian on $\R^3$ that lives in world coordinates and can then be rendered by a standard 3DGS rasterizer.

\subsection{Block decomposition}

Decompose $\Sigma_{4D}$ along the time axis $e_0$:
\begin{equation}\label{eq:block}
\Sigma_{4D} \;=\; \begin{pmatrix} \Sigma_{tt}^{\mathrm{pure}} & c_{\text{world}}^T \\ c_{\text{world}} & \Sigma_{3D} \end{pmatrix}, \qquad \mu \;=\; \begin{pmatrix} v_0 \\ V \end{pmatrix},
\end{equation}
where $\Sigma_{tt}^{\mathrm{pure}} = (\Sigma_{4D})_{00} \in \R$, $c_{\text{world}} = (\Sigma_{4D})_{1:4, 0} \in \R^3$, and $\Sigma_{3D} = (\Sigma_{4D})_{1:4, 1:4} \in \R^{3 \times 3}$. The pure superscript on $\Sigma_{tt}^{\mathrm{pure}}$ emphasizes that this is the unblurred temporal variance, distinct from the blurred quantity $\Sigma_{tt}^{\mathrm{blur}} = \Sigma_{tt}^{\mathrm{pure}} + \sigma_{\text{tmp}}^2$ used elsewhere.

\subsection{Schur conditioning}

Provided $\Sigma_{tt}^{\mathrm{pure}} > 0$, the conditional Gaussian on $\{x_0 = t_0\}$ is $\mathcal{N}(V_{3D}(t_0), \Sigma_{3D}(t_0))$ with
\begin{equation}\label{eq:schur_mean}
\boxed{V_{3D}(t_0) \;=\; V \,+\, c_{\text{world}} \cdot (t_0 - v_0) \,/\, \Sigma_{tt}^{\mathrm{pure}}}
\end{equation}
\begin{equation}\label{eq:schur_cov}
\boxed{\Sigma_{3D}(t_0) \;=\; \Sigma_{3D} \,-\, c_{\text{world}} \, c_{\text{world}}^T \,/\, \Sigma_{tt}^{\mathrm{pure}}}
\end{equation}
This is the standard Gaussian conditioning formula and is exact: no Taylor expansion or other approximation is involved.

\subsection{Rank of the conditioned covariance}

\begin{prop}[Rank-2 after conditioning]\label{prop:rank2}
Provided $\rk(\Sigma_{4D}) = 3$ and $\Sigma_{tt}^{\mathrm{pure}} > 0$, the conditioned spatial covariance $\Sigma_{3D}(t_0)$ has rank exactly 2.
\end{prop}

\begin{proof}
By Proposition~\ref{prop:rank4D}, $\rk(\Sigma_{4D}) \leq 3$ with $\spn(n) \subseteq \ker(\Sigma_{4D})$. Suppose $\rk(\Sigma_{4D}) = 3$, so $\ker(\Sigma_{4D}) = \spn(n)$ and $\img(\Sigma_{4D}) = E_n$ has dimension $3$. The Schur complement $\Sigma_{3D}(t_0)$ is the lower-right block of the inverse of $\Sigma_{4D}$ when restricted to its image, formally:
\[
\Sigma_{3D}(t_0) = \Sigma_{3D} - c_{\text{world}} c_{\text{world}}^T / \Sigma_{tt}^{\mathrm{pure}}.
\]
We argue rank via factorization. Write $\Sigma_{4D} = M M^T$ with $M \in \R^{4 \times 3}$ of rank 3 (this is possible since $\rk(\Sigma_{4D}) = 3$). Let $m_0 \in \R^3$ be the first row of $M$, and let $M' \in \R^{3 \times 3}$ be the lower three rows. Then $\Sigma_{tt}^{\mathrm{pure}} = m_0^T m_0 = \norm{m_0}^2$ and $c_{\text{world}} = M' m_0$, so
\[
\Sigma_{3D}(t_0) \;=\; M' M'^T \,-\, \frac{M' m_0 m_0^T M'^T}{\norm{m_0}^2} \;=\; M' \,\Big(I_3 \,-\, \frac{m_0 m_0^T}{\norm{m_0}^2}\Big)\, M'^T.
\]
The matrix in parentheses is the orthogonal projector onto the hyperplane in $\R^3$ orthogonal to $m_0$, which has rank 2. Since $M$ has rank 3, $M'$ has rank at least 2 (it has at most 1 column in the kernel direction $m_0$, after a basis change), and a routine dimension count shows $\rk(\Sigma_{3D}(t_0)) = 2$ generically. The strict equality holds whenever $\img(M')$ has dimension $\geq 2$ within the orthogonal complement of $m_0$, which is the generic situation.
\end{proof}

\begin{remark}[Geometric meaning of rank 2]
The rank-2 structure has a clean geometric interpretation. The 3-plane $E_n$ in $\R^4$, sliced at time $t_0$, is a 2-plane in $\R^3$ (Proposition~\ref{prop:slice}). The conditioned Gaussian, supported on this slice, is therefore inherently 2-dimensional; its covariance must have rank 2. This is the exact same primitive as the disk-shaped Gaussians of static 3DGS, in which the third scale is set very small to produce a flat splat. Our framework arrives at the disk geometry naturally, without needing an artificial flatness constraint.
\end{remark}

\subsection{Effective opacity and the temporal weight}\label{sec:eff_opacity}

The conditional Gaussian \eqref{eq:schur_mean}--\eqref{eq:schur_cov} captures the spatial \emph{shape} of the splat at time $t_0$, but it does not capture how strongly the Gaussian is \emph{present} at that time. A Gaussian centered at $v_0 = 0$ with temporal extent $\sqrt{\Sigma_{tt}^{\mathrm{pure}}} = 1$ frame should fade out by $t_0 = 100$, and conditioning alone does not produce this fade.

We introduce a temporal saliency weight that modulates the opacity:

\begin{defn}[Temporal weight and effective opacity]\label{defn:wt}
Let $\Sigma_{tt}^{\mathrm{blur}} = \Sigma_{tt}^{\mathrm{pure}} + \sigma_{\text{tmp}}^2$. The temporal weight at frame $t_0$ is
\begin{equation}\label{eq:wt}
w_t \;=\; \exp\!\bigg( -\frac{(t_0 - v_0)^2}{2 \, \Sigma_{tt}^{\mathrm{blur}}} \bigg) \;\in\; (0, 1],
\end{equation}
and the effective opacity is
\begin{equation}\label{eq:alpha_eff}
\boxed{\alpha^{\mathrm{eff}}(t_0) \;=\; \alpha \cdot w_t \;\in\; [0, \alpha].}
\end{equation}
\end{defn}

\begin{remark}[The unnormalized choice]\label{rem:unnormalized}
The weight $w_t$ is the Gaussian \emph{kernel} (peak value $1$), not the Gaussian \emph{density} (which carries a normalization factor $(2 \pi \Sigma_{tt}^{\mathrm{blur}})^{-1/2}$). This is a deliberate modeling choice, not a derivation from the joint Gaussian density of Section~\ref{sec:conditioning}. We make it explicit:
\begin{itemize}
\item The conditioning operation \eqref{eq:schur_mean}--\eqref{eq:schur_cov} produces a properly normalized 2D conditional density in space.
\item The product $\alpha \cdot w_t \cdot \mathcal{N}(\,\cdot\, ; V_{3D}(t_0), \Sigma_{3D}(t_0))$ that we use in the rendering equation \eqref{eq:render} is therefore \emph{not} the joint density $p(y, t_0) = p(y \mid t_0) \cdot p(t_0)$ of the 4D Gaussian, which would carry an extra factor of $(2\pi \Sigma_{tt}^{\mathrm{blur}})^{-1/2}$.
\item We choose to drop the normalization factor in order to keep $\alpha^{\mathrm{eff}}(t_0) \in [0, \alpha] \subseteq [0, 1]$, which is required for the alpha-compositing semantics of the rasterizer.
\item This is consistent with standard 3DGS, which also evaluates spatial Gaussians without their normalization constant; the missing normalization is absorbed by the learned opacity $\alpha$ during training.
\end{itemize}
A consequence is that two Gaussians with different temporal extents but otherwise identical may need quite different $\alpha$ values to represent the same total temporal mass. Density control (Section~\ref{sec:training}) must take this into account when splitting Gaussians.
\end{remark}

\begin{remark}[Why $\Sigma_{tt}^{\mathrm{blur}}$ for $w_t$, $\Sigma_{tt}^{\mathrm{pure}}$ for Schur]
The temporal weight uses $\Sigma_{tt}^{\mathrm{blur}} = \Sigma_{tt}^{\mathrm{pure}} + \sigma_{\text{tmp}}^2$, while the Schur conditioning uses $\Sigma_{tt}^{\mathrm{pure}}$. The reason is geometric: adding $\sigma_{\text{tmp}}^2$ in the Schur denominator would smear the spatial slice with isotropic blur and break the rank-2 structure of Proposition~\ref{prop:rank2}. The temporal weight does not need the rank structure, only a positive denominator, so the modeling floor $\sigma_{\text{tmp}}^2$ is permitted to enter there.

In practice, both denominators are passed through a shared numerical floor $\varepsilon$ before division (Section~\ref{sec:stability}). This is purely a numerical-stability device, distinct from $\sigma_{\text{tmp}}^2$, and prevents NaN at initialization when both quantities can simultaneously approach zero.
\end{remark}

\subsection{Camera awareness, exactly}\label{sec:camera}

A central architectural feature of our framework is that time conditioning happens in \emph{world coordinates}, before any camera transform. We now make precise the statement that this avoids any approximation in the camera trajectory.

Let $R: \R \to \mathrm{SO}(3)$ and $c: \R \to \R^3$ be smooth functions giving the camera rotation and position as a function of time. The world-to-camera transform at time $t$ is $X \mapsto R(t) (X - c(t))$, which is followed by the perspective projection $\pi: \R^3 \to \R^2$.

\begin{prop}[Exactness of world-space conditioning]\label{prop:camera}
The pipeline that produces $(V_{3D}(t_0), \Sigma_{3D}(t_0))$ via \eqref{eq:schur_mean}--\eqref{eq:schur_cov} and then renders with the rasterizer's view matrix $(R(t_0), c(t_0))$ uses no Taylor expansion of $R(t)$ or $c(t)$. The only approximation in the rendered output is the standard EWA linearization of the perspective projection $\pi$ around the projected mean.
\end{prop}

\begin{proof}
The conditioning step \eqref{eq:schur_mean}--\eqref{eq:schur_cov} is a closed-form Schur complement and is exact. The output $(V_{3D}(t_0), \Sigma_{3D}(t_0))$ is in world coordinates and does not depend on $R$ or $c$. The rasterizer's view transform applies the exact pose $(R(t_0), c(t_0))$ at the rendered frame, and its perspective Jacobian $J_\pi$ is evaluated at the camera-space mean $R(t_0)(V_{3D}(t_0) - c(t_0))$. The only linearization in the rasterizer is the substitution of $\pi$ by its first-order Taylor approximation, which is the standard EWA approximation used by every 3DGS implementation. The camera trajectory $R(t), c(t)$ is never linearized in $t$.
\end{proof}

\begin{remark}[Comparison with full pixel-space linearization]
An alternative architecture would linearize the entire map $z \mapsto (\pi(R(z_0)(z_{1:} - c(z_0))), z_0)$ from the 4-dimensional space to pixel-time, producing a $3 \times d$ Jacobian whose construction involves $\dot R(v_0)$ and $\dot c(v_0)$. Conditioning on $t = t_0$ is then performed in pixel-space, on the already-linearized distribution. This architecture (call it Ansatz~B) introduces an additional approximation error proportional to $\norm{\boldsymbol{m}} \cdot \sqrt{\Sigma_{tt}^{\mathrm{pure}}}$, where $\boldsymbol{m} = \dot R(v_0)(V - c(v_0)) - R(v_0) \dot c(v_0)$ is the camera motion vector --- the apparent velocity of the Gaussian center in camera coordinates due solely to camera motion. Concretely, for a Gaussian with temporal extent of $10$ frames and a camera moving at $5$ pixels per frame, the error is on the order of $50$ pixels, far from negligible. Our world-space architecture (Ansatz~A) avoids this entirely.
\end{remark}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Rendering}\label{sec:rendering}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{The full rendering equation}

For each Gaussian $k$ in the scene and each frame $t_0$, our preprocessing produces a triple $(V_{3D, k}(t_0), \Sigma_{3D, k}^{\mathrm{render}}(t_0), \alpha_k^{\mathrm{eff}}(t_0))$ where the spatial covariance is the conditioned-and-lifted matrix
\begin{equation}\label{eq:Sigma_render}
\Sigma_{3D, k}^{\mathrm{render}}(t_0) \;=\; \Sigma_{3D, k}(t_0) \,+\, \sigma_{\text{lift}}^2 I_3.
\end{equation}
This triple, together with a color attribute $c_k$ (constant RGB or evaluated from spherical harmonics), is fed to a standard 3DGS rasterizer. The rasterizer transforms world-space to camera-space via the view matrix $(R(t_0), c(t_0))$, applies the perspective Jacobian $J_\pi$ to obtain a pixel-space covariance $J_\pi R(t_0) \Sigma_{3D}^{\mathrm{render}} R(t_0)^T J_\pi^T$, adds the pixel blur $\sigma_{\text{pix}}^2 I_2$, sorts depths, and front-to-back alpha-composites:
\begin{equation}\label{eq:render}
\boxed{C(y \mid t_0) \;=\; \sum_{k \in \text{sorted}} c_k \cdot \alpha_k^{\mathrm{eff}}(t_0) \cdot p_k(y \mid t_0) \cdot \prod_{j < k} \big(1 - \alpha_j^{\mathrm{eff}}(t_0) \cdot p_j(y \mid t_0)\big).}
\end{equation}
Here $p_k(y \mid t_0)$ is the projected 2D Gaussian density at pixel $y$, evaluated at peak value 1 (i.e., the unnormalized kernel, consistent with Remark~\ref{rem:unnormalized}).

\subsection{Rank progression along the pipeline}

The spatial rank of the primitive evolves through the pipeline as follows.

\begin{center}
\begin{tabular}{lll}
\hline
\textbf{Stage} & \textbf{Object} & \textbf{Rank} \\
\hline
Before conditioning & $\Sigma_{3D}$ in world space (3D block of $\Sigma_{4D}$) & $\leq 3$ \\
After Schur (\S\ref{sec:conditioning}) & $\Sigma_{3D}(t_0)$ & $2$ (a disk in $\R^3$) \\
After 3D lift (\S\ref{sec:rendering}) & $\Sigma_{3D}^{\mathrm{render}}(t_0)$ & $3$ (CUDA EWA needs invertible) \\
After EWA, in 2D pixel space & $J_\pi \Sigma_{3D}^{\mathrm{render}} J_\pi^T + \sigma_{\text{pix}}^2 I_2$ & $2$ (full-rank in 2D) \\
\hline
\end{tabular}
\end{center}

The progression $3 \to 2 \to 3 \to 2$ may seem strange, but each step has a clean interpretation: the conditioning correctly drops rank to recover the disk; the lift is a small numerical accommodation of the rasterizer's invertibility requirement; the pixel-space rank-2 is just dimensionality of the screen.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Numerical Stability}\label{sec:stability}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The pipeline has two genuine sources of numerical fragility, which we address with explicit remedies.

\subsection{Degenerate temporal extent and the topological bridge}\label{sec:degen_temp}

When the plane normal $n$ approaches the time axis (i.e., $n \to \pm e_0$, equivalently $n_{1:} \to 0$), the plane $E_n$ approaches the spatial slice $\{x_0 = 0\}$, and the Gaussian has vanishing extent in the time direction: $\Sigma_{tt}^{\mathrm{pure}} \to 0$. The Schur formulas \eqref{eq:schur_mean}--\eqref{eq:schur_cov} then divide by a quantity approaching zero, and the temporal-weight formula \eqref{eq:wt} encounters $0/0$ at the point $t_0 = v_0$. This regime is not exceptional --- it is exactly the standard initialization of Section~\ref{sec:init}, where $n = e_0$ for every Gaussian.

\begin{prop}[Failure mode of bare formulas]\label{prop:schur_fail}
At $\Sigma_{tt}^{\mathrm{pure}} = 0$ with $\sigma_{\mathrm{tmp}}^2 = 0$ and $t_0 = v_0$, formula \eqref{eq:schur_cov} is undefined and the temporal weight \eqref{eq:wt} evaluates as $0/0$, producing a NaN that propagates through the rendered image and contaminates all gradients in the backward pass.
\end{prop}

\begin{proof}
Direct from \eqref{eq:schur_cov} and \eqref{eq:wt}: both expressions divide by the same quantity, which is identically zero in this configuration.
\end{proof}

\paragraph{Remedy: a shared numerical floor.} We replace every appearance of $\Sigma_{tt}^{\mathrm{pure}}$ and $\Sigma_{tt}^{\mathrm{blur}}$ in a denominator by a soft-clamped version
\begin{equation}\label{eq:schur_clamp}
\widetilde{\Sigma}_{tt}^{\mathrm{pure}} = \sqrt{(\Sigma_{tt}^{\mathrm{pure}})^2 + \varepsilon^2}, \qquad \widetilde{\Sigma}_{tt}^{\mathrm{blur}} = \sqrt{(\Sigma_{tt}^{\mathrm{blur}})^2 + \varepsilon^2},
\end{equation}
for a small numerical floor $\varepsilon$ (default $\varepsilon = 10^{-8}$). For $\Sigma_{tt} \gg \varepsilon$ these are indistinguishable from the unclamped quantities, while for $\Sigma_{tt} \to 0$ they remain positive and smooth, keeping all derivatives finite. The floor $\varepsilon$ is a pure numerical guard, distinct from the modeling parameter $\sigma_{\text{tmp}}^2$.

\paragraph{The clamp as a topological bridge.} The clamp does more than prevent NaN: it provides a smooth interpolation between two qualitatively different geometric regimes.

\begin{prop}[Smooth rank transition]\label{prop:bridge}
Let $n$ be parameterized in a neighborhood of $e_0$ by a small tilt angle $\theta$, so that $n_{1:} = O(\theta)$. Generically:
\begin{enumerate}
\item $\Sigma_{tt}^{\mathrm{pure}} = O(\theta^2)$ as $\theta \to 0$;
\item $c_{\mathrm{world}} = O(\theta)$ as $\theta \to 0$.
\end{enumerate}
Consequently the un-clamped Schur correction $c_{\mathrm{world}} c_{\mathrm{world}}^T / \Sigma_{tt}^{\mathrm{pure}} = O(1)$ is finite as $\theta \to 0$ and produces a well-defined rank-2 disk. With the clamp \eqref{eq:schur_clamp}, the correction degrades to $O(\theta^2 / \varepsilon)$ when $\theta^2 \ll \varepsilon$. In this regime $\Sigma_{3D}(t_0) \to \Sigma_{3D}$, which has rank up to $3$. As $\theta^2$ grows past $\varepsilon$, the full Schur correction returns and the splat collapses smoothly to the rank-$2$ disk.
\end{prop}

\begin{proof}
For a unit vector $n = (n_0, n_{1:})$ with $n_{1:}$ a small perturbation of zero, write $n_0 = \sqrt{1 - \norm{n_{1:}}^2}$ and $\theta = \norm{n_{1:}}$. The projector $P_n = I - n n^T$ has time-row $(1 - n_0^2, -n_0 n_{1:}^T) = (\theta^2, -n_0 n_{1:}^T)$, so $L_n = P_n L$ has its first row of order $O(\theta^2 \cdot 1, \theta \cdot \theta) = O(\theta^2)$ entrywise; hence $\Sigma_{tt}^{\mathrm{pure}} = \norm{(L_n)_0}^2 = O(\theta^2)$. The cross-block $c_{\mathrm{world}} = (L_n)_{1:} (L_n)_0^T$ couples a column of order $O(1)$ with a column of order $O(\theta)$, giving $O(\theta)$. The remaining statements follow by direct substitution.
\end{proof}

\begin{remark}[Implications]\label{rem:bridge}
This is a desirable property: a Gaussian initialized at $n = e_0$ behaves indistinguishably from a static 3DGS Gaussian at iteration $0$ (rank-$3$ blob, no temporal evolution), and only develops genuine temporal disk geometry as the optimizer tilts $n$ past $\theta \sim \sqrt{\varepsilon}$. The clamp acts as a smooth bridge between the static-3DGS regime and the dynamic disk regime, with a controlled crossover at $\theta = \sqrt{\varepsilon}$. For the default $\varepsilon = 10^{-8}$, this crossover occurs at $\theta \approx 10^{-4}$~rad --- a negligible departure from $e_0$ that is crossed within the first few optimizer steps for any genuinely time-varying scene content. In particular: the pipeline can be initialized as a static 3DGS scene and evolves continuously into the full dynamic representation. There is no discontinuity at the boundary.
\end{remark}

\begin{remark}[Why not just use $\Sigma_{tt}^{\mathrm{pure}} + \sigma_{\text{tmp}}^2$ everywhere?]
A simpler design uses $\Sigma_{tt}^{\mathrm{pure}} + \sigma_{\text{tmp}}^2$ in both Schur and $w_t$, with $\sigma_{\text{tmp}}^2 > 0$ acting as a single combined modeling-and-numerical floor. This eliminates the need for $\varepsilon$ but couples the two roles: $\sigma_{\text{tmp}}^2$ then must be set large enough for numerical stability, which forces every Gaussian to have a minimum visible duration --- not a property one always wants. The two-floor design ($\varepsilon$ for stability, $\sigma_{\text{tmp}}^2$ for modeling) keeps the roles separated.
\end{remark}

\subsection{Anisotropic spatial covariance}

A second concern is that $\Sigma_{3D}(t_0)$, while always rank 2, may have a very small ratio of its two nonzero eigenvalues --- a near-degenerate disk that is essentially a needle. After the rank lift \eqref{eq:Sigma_render}, the smallest eigenvalue of $\Sigma_{3D}^{\mathrm{render}}(t_0)$ is at most $\sigma_{\text{lift}}^2$, but the second-smallest eigenvalue may also be tiny if the disk is anisotropic, leading to a needle-shaped splat regardless of the original geometry.

This is an attenuated form of the streaking pathology that affects the legacy 2-plane (i.e., $\Gr(2,4)$) framework. In our setting it is less severe because the rank-2 disk has two nonzero eigenvalues by construction, but it can still produce visual artifacts.

\paragraph{Remedy.} Density control (Section~\ref{sec:training}) detects Gaussians with extreme anisotropy --- via the ratio $\lambda_1 / \lambda_2$ of the nonzero eigenvalues of $\Sigma_{3D}(t_0)$ --- and splits or prunes them.

\subsection{The rank lift $\sigma_{\text{lift}}^2$}

We have noted that $\sigma_{\text{lift}}^2$ exists to make the rasterizer's CUDA EWA kernel invertible. The scale of $\sigma_{\text{lift}}$ should be substantially smaller than the typical scale of the disk, so that it does not visibly fatten the splat in the rendered image. In typical scenes (with scene scale measured in meters), $\sigma_{\text{lift}}^2 = 10^{-4}$ corresponds to roughly $1$~cm of radius, well below pixel resolution at standard rendering distances. We treat $\sigma_{\text{lift}}^2$ as a fixed numerical parameter, not a learned one.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Gradients}\label{sec:gradients}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The training loop optimizes the per-Gaussian parameters $(n_{\text{raw}}, L, \mu, \alpha, \text{color})$ together with the global blur scalars by gradient descent on a video reconstruction loss. We describe how gradients propagate through each component of the pipeline.

\subsection{Autograd-handled chain}\label{sec:autograd}

The map from raw parameters to the rasterizer input is a composition of standard PyTorch operations:
\begin{enumerate}
\item Normalize the plane normal: $n = n_{\text{raw}} / \norm{n_{\text{raw}}}$.
\item Project the factor: $L_n = L - n (n^T L)$.
\item Form $\Sigma_{4D} = L_n L_n^T$.
\item Block-decompose $\Sigma_{4D}$ into $\Sigma_{tt}^{\mathrm{pure}}, c_{\text{world}}, \Sigma_{3D}$ (slicing).
\item Apply the Schur step \eqref{eq:schur_mean}--\eqref{eq:schur_cov} with the clamped denominator \eqref{eq:schur_clamp}.
\item Compute the temporal weight $w_t$ \eqref{eq:wt} and effective opacity \eqref{eq:alpha_eff}.
\item Add the rank lift: $\Sigma_{3D}^{\mathrm{render}} = \Sigma_{3D}(t_0) + \sigma_{\text{lift}}^2 I_3$.
\item Pass $(V_{3D}(t_0), \Sigma_{3D}^{\mathrm{render}}(t_0), \alpha^{\mathrm{eff}})$ and color to the rasterizer.
\end{enumerate}
All steps 1--7 are pure PyTorch tensor operations, and step 8 is a standard 3DGS rasterizer with a documented backward pass providing $\partial \mathcal{L} / \partial V_{3D}^{\mathrm{render}}$, $\partial \mathcal{L} / \partial \Sigma_{3D}^{\mathrm{render}}$, $\partial \mathcal{L} / \partial \alpha^{\mathrm{eff}}$, and $\partial \mathcal{L} / \partial \text{color}$. PyTorch autograd composes these into gradients with respect to all raw parameters.

\subsection{Riemannian step on $S^3$}\label{sec:riemannian}

The plane normal $n$ lives on the manifold $S^3 / \{\pm 1\}$. Two complementary strategies are used.

\paragraph{Implicit (renormalize-in-forward).} The forward pass renormalizes $n_{\text{raw}}$ on every iteration: $n = n_{\text{raw}} / \norm{n_{\text{raw}}}$. Autograd through this normalization automatically produces a gradient on $n_{\text{raw}}$ that is tangent to $S^3$ (in the limit; up to floating-point noise). This is the simplest and most common approach in practice and requires no manual intervention.

\paragraph{Explicit (Riemannian projection).} For tight numerical control, after autograd produces a Euclidean gradient $\nabla_n \mathcal{L} \in \R^4$, project onto the tangent space $T_n S^3$ via
\begin{equation}\label{eq:riemannian}
\mathrm{grad}_{S^3} \mathcal{L} \big|_n \;=\; \nabla_n \mathcal{L} \,-\, \ip{n, \nabla_n \mathcal{L}} \, n,
\end{equation}
and update by retraction: $n \leftarrow (n - \eta \cdot \mathrm{grad}_{S^3} \mathcal{L}) / \norm{n - \eta \cdot \mathrm{grad}_{S^3} \mathcal{L}}$ for learning rate $\eta$. The two-fold cover $S^3 \to S^3 / \{\pm 1\}$ does not affect the local optimization since both $n$ and $-n$ are equivalent representatives of the same plane; either is allowed.

The implicit method is preferred for routine training; the explicit method is used as a numerical-conditioning aid when training becomes unstable.

\subsection{Gauge equivalence of $L$}\label{sec:gauge}

As discussed in Section~\ref{sec:Ldof}, the factor $L$ has 6 redundant directions: 3 in the $n$-component (annihilated by $P_n$) and 3 in the right $\mathrm{O}(3)$-action on columns. The optimizer's gradient on $L$ vanishes in directions of type 1 (since $P_n$ kills those components) and is invariant to changes within the gauge orbit of type 2 (since $\Sigma_{4D}$ is). No explicit handling is required; the optimizer simply settles into one orbit representative.

\subsection{Other parameters}

The mean $\mu \in \R^4$ is unconstrained (Proposition~\ref{prop:mu_dof}), as is the factor $L \in \R^{4 \times 3}$. The opacity $\alpha$ is stored as a logit and passed through a sigmoid. Color is either a sigmoid-RGB triple or unconstrained spherical harmonic coefficients. None of these require manifold projection.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Initialization and Training}\label{sec:training}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{Input data and preprocessing}

The framework accepts as input a video sequence with known camera poses $(R(t), c(t))$ at each frame. Camera poses are typically obtained from structure-from-motion or visual-inertial odometry as a preprocessing step and are not optimized jointly with the scene. Optionally, a per-frame point cloud (from depth sensors or stereo) and a per-frame scene flow estimate may be provided.

\begin{remark}[Timestamp normalization is required]\label{rem:timescale}
The factor $L \in \R^{4 \times 3}$ couples time and space additively, and the projector $P_n = I - n n^T$ uses the standard Euclidean inner product on $\R^4$. Both implicitly assume that time and space share commensurable units. For raw input where $t$ is measured in frame indices (e.g.\ $0, 1, \ldots, 300$) and $X$ is measured in metric scene units (e.g.\ $X \in [-1, 1]^3$), this assumption fails: the spacetime metric becomes wildly anisotropic, and gradients on the time component of $n$ are scaled by orders of magnitude relative to the spatial components. The Riemannian projection on $S^3$ (\S\ref{sec:riemannian}) inherits this distortion.

We therefore require timestamp normalization as a preprocessing step: rescale the temporal axis so that the full sequence spans a range comparable to the spatial bounding box, e.g.\ $t \in [-1, 1]$ uniformly across the sequence. This is a simple change of units, but skipping it produces optimization pathologies that look like hyperparameter problems.
\end{remark}

\subsection{Initialization}\label{sec:init}

The initialization places one Gaussian per seed point and chooses the plane orientation $n$ for each. Several strategies are admissible.

\paragraph{Default: spatial slice initialization.} The simplest and most robust choice is $n = e_0$ for all Gaussians, so each plane $E_n$ is the spatial slice $\{x_0 = 0\}$. This is the ``static'' initialization: the plane has no temporal tilt, and the Gaussian behaves as a static 3DGS primitive. The optimizer is free to tilt $n$ during training to capture motion.

\paragraph{Mean.} For each seed point $X_k$ at frame $t_k$, set $\mu_k = (t_k, X_k)$.

\paragraph{Factor.} Initialize $L_k$ as a small isotropic factor: $L_k = \rho \cdot \widetilde{L}_k$ where $\widetilde{L}_k \in \R^{4 \times 3}$ has i.i.d.\ standard normal entries and $\rho$ is set so that $\sqrt{\tr(\Sigma_{4D, k}) / 3}$ matches the local point cloud spacing. After projection $P_n L_k$ and (with $n = e_0$) the time row of $L_n$ vanishes, reducing the effective covariance to a 3D spatial Gaussian.

\paragraph{Opacity.} Initialize $\alpha_k = 0.1$ (small, to be raised by the optimizer when the Gaussian is useful).

\paragraph{Optional: flow-aware initialization.} If scene flow $\dot{X}_k$ is available, initialize the time component of $n$ to encode the apparent velocity. By Proposition~\ref{prop:slice}, the slice plane has velocity vector $-n_0 \, n_{1:} / \norm{n_{1:}}^2 \in \R^3$. To match a flow vector with unit direction $\hat u$ and speed $s$, choose $n_{1:} \propto \hat u$ and $n_0 = -s \norm{n_{1:}}$, then renormalize $n$. This is a more aggressive initialization that may improve convergence on highly dynamic scenes at the cost of robustness.

\subsection{Loss}\label{sec:loss}

The training loss is a sum of an L1 reconstruction term, a perceptual LPIPS term, and a temporal LPIPS term:
\begin{equation}\label{eq:loss}
\mathcal{L} \;=\; \lambda_1 \mathcal{L}_1 \,+\, \lambda_{\mathrm{LPIPS}} \mathcal{L}_{\mathrm{LPIPS}} \,+\, \lambda_{\mathrm{TLPIPS}} \mathcal{L}_{\mathrm{TLPIPS}}.
\end{equation}
The L1 term is the per-pixel mean absolute error between rendered and ground-truth frames. The LPIPS term \cite{LPIPS} is the perceptual metric applied to each frame. The temporal LPIPS term applies the LPIPS metric to frame differences:
\[
\mathcal{L}_{\mathrm{TLPIPS}}(\hat V, V) \;=\; \mathcal{L}_{\mathrm{LPIPS}}(\hat V_{1:T} - \hat V_{0:T-1}, \, V_{1:T} - V_{0:T-1}),
\]
penalizing temporal flicker in the rendered video. Default weights: $\lambda_1 = 0.8$, $\lambda_{\mathrm{LPIPS}} = 0.1$, $\lambda_{\mathrm{TLPIPS}} = 0.1$.

\begin{remark}[Caveat on temporal LPIPS]
LPIPS is computed via VGG/AlexNet features trained on natural images. Frame differences $\hat V - \hat V_{\mathrm{prev}}$ are far from natural --- they are predominantly near-zero signals with sparse motion edges --- so the resulting feature distances should be interpreted with caution. The temporal LPIPS term does empirically suppress flicker in our experiments, but a principled alternative based on optical-flow warping ($\norm{\mathcal{W}_{f_t}(\hat V_t) - \hat V_{t+1}}$ for an estimated flow $f_t$) may be preferable for sequences with strong motion and is left for future investigation.
\end{remark}

\subsection{Density control}\label{sec:density}

Following standard 3DGS practice, every $K$ iterations (default $K = 100$) the population of Gaussians is adjusted by clone, split, and prune operations. The criteria are adapted from 3DGS to account for the spacetime nature of our primitive.

\paragraph{Prune (remove).} Remove Gaussian $k$ if any of the following hold:
\begin{itemize}
\item Base opacity $\alpha_k$ falls below a threshold $\alpha_{\min}$ (default $\alpha_{\min} = 0.005$).
\item The largest eigenvalue of $\Sigma_{3D, k}$ exceeds a scene-scale threshold (Gaussian is too large in space).
\item The temporal extent $\Sigma_{tt, k}^{\mathrm{pure}}$ exceeds the full sequence length (Gaussian is unhelpfully temporally diffuse).
\end{itemize}

\paragraph{Clone (duplicate at same location).} If the spatial gradient $\partial \mathcal{L} / \partial V_k$ is large (above a threshold) and the Gaussian is small, clone it: produce a copy at the same $\mu$ with perturbed $L$. This is the standard 3DGS clone heuristic; the temporal axis is treated symmetrically.

\paragraph{Split (subdivide into smaller Gaussians).} If the gradient is large and the Gaussian is large, split into two: place new means at $\mu \pm \delta \cdot v_{\max}$ where $v_{\max}$ is the eigenvector of $\Sigma_{4D}$ with largest eigenvalue. The factor $L$ is scaled down by $1/\sqrt{2}$ for each child to preserve total mass.

\paragraph{Temporal split (specific to our framework).} If the Gaussian's temporal extent $\sqrt{\Sigma_{tt}^{\mathrm{pure}}}$ exceeds a fraction (default $25\%$) of the sequence length and the gradient is large, split along the temporal axis specifically. This addresses the smallness condition discussed in Proposition~\ref{prop:camera}: a temporally diffuse Gaussian under a fast-moving camera benefits from being subdivided into temporally narrow components.

\paragraph{Anisotropy split.} If the ratio of nonzero eigenvalues of $\Sigma_{3D}(t_0)$ exceeds a threshold (default $20$) at any frame in the training set, split the Gaussian along the long axis of the disk. This addresses the needle-shaped pathology of Section~\ref{sec:stability}.

\subsection{Optimizer and schedule}

Parameters are updated with Adam, with separate learning rates for different parameter groups:
\begin{center}
\begin{tabular}{ll}
\hline
\textbf{Parameter} & \textbf{Learning rate} \\
\hline
$n_{\text{raw}}$ & $10^{-3}$ \\
$L$ & $5 \cdot 10^{-3}$ \\
$\mu$ & $10^{-4}$ (spatial), $10^{-3}$ (time) \\
$\alpha$ (logit) & $5 \cdot 10^{-2}$ \\
SH coefficients & $2.5 \cdot 10^{-3}$ \\
\hline
\end{tabular}
\end{center}
A standard exponential learning-rate decay is applied. Density control is run every $100$ iterations after a $500$-iteration warm-up phase.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Implementation Recipe}\label{sec:algorithm}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We give a complete per-iteration pseudocode for the forward pass at frame $t_0$.

\begin{lstlisting}[caption={Forward pass at frame $t_0$ (PyTorch).},label={lst:forward}]
# Per-Gaussian raw parameters (batched over k):
#   n_raw       : [K, 4]      plane normal, unconstrained
#   L           : [K, 4, 3]   Cholesky-like factor, unconstrained
#   mu          : [K, 4]      mean in spacetime
#   alpha_logit : [K]         opacity logit
#   color       : [K, 3]      or [K, sh_dim, 3] for SH
#
# Globals:
#   sigma_tmp_sq, sigma_lift_sq   (variances)
#   eps                           (Schur denominator floor)

# 1. Normalize plane normal
n = n_raw / n_raw.norm(dim=-1, keepdim=True)   # [K, 4]

# 2. Project the factor: L_n = L - n (n^T L)
nL = n.unsqueeze(-1) @ (n.unsqueeze(-2) @ L)   # [K,4,1] @ [K,1,3] = [K,4,3]
L_n = L - nL                                    # [K, 4, 3]

# 3. Build 4D covariance
Sigma_4D = L_n @ L_n.transpose(-1, -2)   # [K, 4, 4]

# 4. Block decompose along time axis
Sigma_tt_pure = Sigma_4D[:, 0, 0]         # [K]
c_world       = Sigma_4D[:, 1:, 0]        # [K, 3]
Sigma_3D      = Sigma_4D[:, 1:, 1:]       # [K, 3, 3]
v0 = mu[:, 0]                              # [K]
V  = mu[:, 1:]                             # [K, 3]

# 5. Soft-clamped Schur denominator (stability)
Sigma_tt_clamped = torch.sqrt(Sigma_tt_pure ** 2 + eps ** 2)

# 6. Schur conditioning
dt = t0 - v0                                # [K]
V_3D_t   = V + c_world * (dt / Sigma_tt_clamped).unsqueeze(-1)
Sigma_3D_t = Sigma_3D - (
    c_world.unsqueeze(-1) @ c_world.unsqueeze(-2)
) / Sigma_tt_clamped.view(-1, 1, 1)

# 7. Temporal weight and effective opacity
#    The blur denominator is also clamped with eps: at default initialization
#    (n = e_0 -> Sigma_tt_pure = 0) and zero modeling floor (sigma_tmp_sq = 0),
#    the un-clamped denominator would be exactly zero and dt = 0 yields 0/0 = NaN.
Sigma_tt_blur_clamped = torch.sqrt(
    (Sigma_tt_pure + sigma_tmp_sq) ** 2 + eps ** 2
)
w_t = torch.exp(-(dt ** 2) / (2.0 * Sigma_tt_blur_clamped))
alpha = torch.sigmoid(alpha_logit)
alpha_eff = alpha * w_t

# 8. Rank lift (CUDA EWA invertibility)
Sigma_3D_render = Sigma_3D_t + sigma_lift_sq * torch.eye(3, device=L.device)

# 9. Cull Gaussians with negligible contribution
visible = alpha_eff > 1.0 / 255.0
# ... apply visible mask to all per-Gaussian tensors ...

# 10. Pass to standard 3DGS rasterizer (e.g. diff-gaussian-rasterization)
image = rasterize(
    means3D       = V_3D_t,
    cov3D_precomp = Sigma_3D_render.flatten_to_6_unique(),
    opacities     = alpha_eff,
    sh_or_colors  = color,
    viewmatrix    = view_matrix_at(t0),
    projmatrix    = proj_matrix_at(t0),
)
\end{lstlisting}

The rasterizer expects the 3D covariance as a 6-element vector (the upper-triangular part of the symmetric $3 \times 3$); the conversion is straightforward.

\paragraph{Backward pass.} All steps above are pure PyTorch and the rasterizer provides a backward pass. PyTorch autograd produces gradients for $n_{\text{raw}}, L, \mu, \alpha_{\text{logit}}, \text{color}$ automatically.

\paragraph{Computational cost.} Steps 1--8 add a constant overhead per Gaussian, dominated by the $4 \times 3$ matrix multiplications and the $4 \times 4$ symmetric eigendecomposition (only the block decomposition, which is just slicing). The dominant cost remains the rasterizer, which is unchanged from standard 3DGS. In practice the preprocessing overhead is below 5\% of total per-frame cost.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Comparison with Standard 3DGS}\label{sec:dof}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{Side-by-side}

\begin{center}
\begin{tabular}{lcc}
\hline
& \textbf{3DGS (static)} & \textbf{Grassmann ($\Gr(3,4)$ projector)} \\
\hline
Primitive & 3D ellipsoid in $\R^3$ & 3-plane in $\R^4$ (sliced disk in $\R^3$) \\
Plane parameter & --- & $n \in S^3 / \{\pm 1\}$, 3 DOF \\
Mean & $\mu \in \R^3$, 3 DOF & $\mu \in \R^4$, 4 DOF \\
Spatial covariance & $\PSD_3$, 6 DOF & $L \in \R^{4 \times 3}$ mod gauge, 6 DOF \\
\textbf{Geometry total} & \textbf{9} & \textbf{13} \\
\hline
Opacity & 1 DOF & 1 DOF \\
Color (SH degree $\ell$) & $3 (\ell+1)^2$ & $3 (\ell+1)^2$ \\
\hline
Temporal evolution & none & encoded in $n$ tilt, $c_{\text{world}}$ \\
Camera motion & view matrix & view matrix at exact $t_0$ pose \\
Rasterizer & standard 3DGS & \textbf{same, unmodified} \\
Custom CUDA & --- & \textbf{none} \\
\hline
\end{tabular}
\end{center}

The framework adds 4 geometry DOFs over 3DGS (13 vs.\ 9): 3 for the plane orientation $n$ in spacetime, and 1 for the temporal mean $v_0$. The 6 spatial-covariance DOFs are unchanged: the projector form parameterizes the same 6-dimensional family of $3 \times 3$ PSD matrices, expressed in plane-aware coordinates.

\subsection{Relation to static 3DGS via the rank bridge}

The framework is a strict generalization of static 3DGS, but the relationship is more subtle than ``set $n = e_0$ and freeze.'' Two ingredients connect them.

\emph{Spatially, the bridge of Proposition~\ref{prop:bridge}} guarantees that as the plane normal $n$ tilts away from $e_0$, the spatial covariance $\Sigma_{3D}(t_0)$ transitions smoothly from rank $3$ (a blob, as in static 3DGS) to rank $2$ (a moving disk, the proper dynamic primitive). Within the small window $\theta < \sqrt{\varepsilon}$ around $n = e_0$, each Gaussian renders as a rank-$3$ blob spatially indistinguishable from a static 3DGS primitive. Initialization at $n = e_0$ therefore starts the optimization in the static-3DGS regime, and dynamic geometry is unlocked continuously as $n$ tilts.

\emph{Temporally,} however, the framework retains the per-Gaussian temporal saliency $w_t$ regardless of $n$. To make every Gaussian visible at every frame --- the static-3DGS behavior --- one must additionally set $\sigma_{\text{tmp}}^2$ large enough that $w_t \approx 1$ over the full sequence. Without this, even with $n = e_0$, a Gaussian centered at temporal mean $v_0$ contributes only near frames $t_0 \approx v_0$. This is the price of the unified architecture: the temporal aspect is genuinely new geometry, not a hidden hyperparameter.

In summary: the framework is best viewed as a continuous family of representations interpolating between static-3DGS-like primitives (rank-$3$, narrow temporal extent at small $\theta$) and dynamic disks (rank-$2$, possibly broad in time at large $\theta$), controlled by the optimizer's chosen direction in parameter space.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Open Questions}\label{sec:open}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We close with three directions for further work that are not addressed in the present draft.

\subsection{Initialization without depth}

The default initialization of Section~\ref{sec:init} assumes a per-frame point cloud, typically obtained from depth or stereo. For pure monocular RGB input without depth, an initial point cloud must be extracted (e.g.\ from monocular depth estimation networks), and the temporal alignment of seed points across frames must be guessed. The framework is agnostic to this choice but performance depends sensitively on the initial point cloud quality. A principled study of initialization schemes for monocular RGB input is open.

\subsection{The plane structure as an inductive bias}

The framework restricts the geometric primitive to a 3-plane in spacetime, equivalently a translating 2-plane in $\R^3$ (Proposition~\ref{prop:slice}). This is an inductive bias: rotating disks or accelerating disks cannot be exactly represented by a single primitive and must be approximated by a temporally piecewise sequence of primitives, glued together by density control. A natural extension would be to allow per-Gaussian time-varying plane normals $n(t)$, e.g.\ via a low-rank temporal basis. The trade-off between expressiveness and parameter count requires investigation.

\subsection{Rademacher complexity and amenability}

A recent line of work connects the Rademacher complexity of a function class to the amenability of an associated group action. The action of the rigid motion group $\mathrm{SE}(3)$ on $\R^3$ is non-amenable, but its action on the space of affine lines (or planes) in $\R^3$ is amenable. This suggests that line- or plane-based scene representations may admit tighter generalization bounds than point-based representations. A precise statement of such a bound, in the setting of Grassmann splatting, is an attractive open question.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\appendix
\section{Schur Rank Lemma}\label{app:schur}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We give a self-contained proof of the rank statement used in Proposition~\ref{prop:rank2}.

\begin{lem}[Schur rank drop]
Let $S \in \PSD_{1+d}$ be a positive semidefinite matrix of rank $r$, blocked as
\[
S = \begin{pmatrix} a & b^T \\ b & C \end{pmatrix}
\]
with $a > 0$, $b \in \R^d$, $C \in \PSD_d$. Then $\rk(C - b b^T / a) = r - 1$.
\end{lem}

\begin{proof}
Since $S$ is PSD of rank $r$, write $S = M M^T$ with $M \in \R^{(1+d) \times r}$ of rank $r$. Let $m_0 \in \R^r$ be the first row of $M$ and $M' \in \R^{d \times r}$ be the lower $d$ rows. Then $a = m_0 m_0^T = \norm{m_0}^2$, $b = M' m_0$, and $C = M' M'^T$. Substituting:
\[
C - b b^T / a \;=\; M' M'^T - M' m_0 m_0^T M'^T / \norm{m_0}^2 \;=\; M' \big( I_r - m_0 m_0^T / \norm{m_0}^2 \big) M'^T.
\]
The matrix $\Pi = I_r - m_0 m_0^T / \norm{m_0}^2$ is the orthogonal projector onto $\spn(m_0)^\perp \subset \R^r$, of rank $r - 1$. Choose an orthonormal basis $\{u_1, \ldots, u_{r-1}\}$ of $\spn(m_0)^\perp$, write $\Pi = U U^T$ with $U \in \R^{r \times (r-1)}$ orthogonal in columns. Then $C - bb^T/a = (M' U)(M' U)^T$, a rank-$\leq (r-1)$ Gram matrix.

For the lower bound, we show that $M'$ restricted to $\spn(m_0)^\perp$ is injective. Let $v \in \spn(m_0)^\perp \subset \R^r$ with $M' v = 0$. Then
\[
M v = \begin{pmatrix} m_0^T v \\ M' v \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} = 0,
\]
where the first entry vanishes because $v \in \spn(m_0)^\perp$ means $m_0^T v = 0$, and the second by hypothesis. Since $M$ has rank $r$ its kernel is trivial, so $v = 0$. Hence $M'\restriction_{\spn(m_0)^\perp}$ is injective on the $(r{-}1)$-dimensional space $\spn(m_0)^\perp$, giving $\rk(M' U) = r - 1$ and therefore $\rk(C - bb^T/a) = r - 1$.
\end{proof}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{thebibliography}{99}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\bibitem{3DGS}
B.~Kerbl, G.~Kopanas, T.~Leimk\"uhler, G.~Drettakis,
\textit{3D Gaussian splatting for real-time radiance field rendering},
ACM Transactions on Graphics 42(4), 2023.

\bibitem{EWA}
M.~Zwicker, H.~Pfister, J.~van~Baar, M.~Gross,
\textit{EWA volume splatting},
IEEE Visualization, 2001.

\bibitem{LPIPS}
R.~Zhang, P.~Isola, A.~A.~Efros, E.~Shechtman, O.~Wang,
\textit{The unreasonable effectiveness of deep features as a perceptual metric},
CVPR, 2018.

\end{thebibliography}

\end{document}