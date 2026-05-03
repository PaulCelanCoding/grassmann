\documentclass[11pt]{amsart}
\usepackage{amssymb,amsmath,amsfonts,amsthm,mathtools}
\usepackage[top=1in,margin=1.2in,bottom=1in]{geometry}
\usepackage[colorlinks=true,linkcolor=blue]{hyperref}

\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}
\DeclarePairedDelimiter{\norm}{\lVert}{\rVert}
\newcommand{\ip}[1]{\langle #1 \rangle}
\newcommand{\HH}{\mathcal{H}}

\theoremstyle{plain}
\newtheorem{prop}{Proposition}
\newtheorem{lem}[prop]{Lemma}
\newtheorem{cor}[prop]{Corollary}
\newtheorem{theo}[prop]{Theorem}
\theoremstyle{remark}
\newtheorem{remark}[prop]{Remark}
\theoremstyle{definition}
\newtheorem{defn}[prop]{Definition}

\begin{document}
\title{Jacobian of the Grassmannian Rendering Pipeline:\\
From Canonical Bundle to Pixel$\times$Time Space}
\author{Technical Note for the Grassmann Framework}
\date{v6 --- camera-aware analysis with full comparison theorem}
\maketitle

\noindent\textbf{Changelog}
\begin{description}
\item[v1] Initial derivation: basis for $E_{p,q}$, Jacobian factorization, projection to $\mathbb{R}^2$ (spatial only).
\item[v2] Corrected target space to $\mathbb{R}^2 \times \mathbb{R}$ (pixel $\times$ time). Added $J_{\mathrm{time}}$, conditioning on $t_0$, full $3\times 2$ Jacobian.
\item[v3] Corrected temporal opacity: joint density = conditional $\times$ marginal. Added effective opacity $\alpha_k^{\mathrm{eff}}(t_0)$, alpha compositing with transmittance.
\item[v4] Added high-level overview and key results summary at top of document.
\item[v5] Incorporated reviewer feedback: (1) 3D-lifted conditioning in world space (fixes view-matrix bug), (2) rank-1 after time conditioning (corrects Remark~11), (3) unnormalized temporal weight (prevents opacity explosion). Acknowledgments added.
\item[v6] Full camera-aware analysis: Case~A (static camera, exact equivalence with 3D-lifted), Case~B (dynamic camera, camera motion vector $\mathbf{m}$, Jacobian $J_{\mathrm{cam}}$). Theorem~\ref{thm:comparison} with complete proof: quantifies the error difference between 3D-lifted (Ansatz~A) and full linearization (Ansatz~B), showing Ansatz~A is strictly more accurate with error $O(\norm{\boldsymbol{\eta}}\sqrt{\sigma_{\beta\beta}})$.
\end{description}
\bigskip

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section*{Overview: The Problem and Our Solution}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection*{The setting}
The Grassmann Framework represents a dynamic 3D scene using Gaussians that live in the canonical bundle $E \to \mathrm{Gr}^+(4,2) \cong S^2 \times S^2$ over the oriented Grassmannian. Each Gaussian encodes a line in 3D space together with its temporal evolution, using only 9 geometric degrees of freedom --- the same count as standard 3D Gaussian Splatting, but with time ``for free.''

\subsection*{The core problem}
To render a frame, we need to evaluate how much each Gaussian contributes to each pixel. The original paper defines this via an integral (the marginal distribution):
\[
p_k(y) = \int_{E_{p,q}} \mathcal{N}(y;\; P(z),\; \sigma_k^2 I)\;\mathcal{N}(z;\; z_k,\; \Sigma_k)\; dz.
\]
This integral has \textbf{no closed-form solution} because the projection $P$ contains a perspective division (division by depth $Z$), making it a rational --- not affine --- function. If $P$ were affine, the integral would be a standard Gaussian convolution with an elementary closed-form result. But it is not, and the integral must be evaluated billions of times per second for real-time rendering, ruling out numerical approaches such as Monte Carlo integration.

\subsection*{Our approach}
We resolve this in three steps:

\textbf{Step 1: Linearize $P$.} We replace $P(z)$ by its first-order Taylor expansion around the Gaussian mean $z_k$:
\[
P(z) \approx P(z_k) + J\cdot(z - z_k).
\]
This is the standard EWA splatting technique (Zwicker et al., 2001), used by every existing 3DGS implementation. With this substitution, the integral becomes a Gaussian convolution and evaluates to:
\[
p_k(y, t) \approx \mathcal{N}\!\left(\begin{pmatrix} y \\ t \end{pmatrix};\; P(z_k),\; J\,\Sigma_k\, J^T + \sigma_k^2 I\right).
\]

\textbf{Step 2: Compute $J$ explicitly.} The main technical contribution of this note is an explicit, clean formula for the $3\times 2$ Jacobian $J$. We first construct an orthogonal basis $\{e_1 = p+q,\; e_2 = 1-pq\}$ for the canonical plane $E_{p,q}$ (Proposition~\ref{prop:basis}), and then show that $J$ factorizes as:
\[
J_{\mathrm{full}} = \begin{pmatrix} J_{\mathrm{persp}} \cdot J_{\mathrm{embed}} \\ J_{\mathrm{time}} \end{pmatrix}
\]
where $J_{\mathrm{persp}}$ is the standard $2\times 3$ perspective Jacobian (already implemented in every 3DGS rasterizer), $J_{\mathrm{embed}}$ is a $3\times 2$ matrix built from $p+q$ and $p\times q$ (the spatial embedding of the canonical plane), and $J_{\mathrm{time}} = (0,\;\sqrt{(1+p\cdot q)/2}\,)$ extracts the time component. We analyze two cases (Section~\ref{sec:camera_aware}): for a \emph{static camera} (Case A), the Jacobian acquires a rotation factor $R_0$ and the analysis is exact; for a \emph{dynamic camera} (Case B), an additional camera motion vector $\mathbf{m}$ appears in the $\beta$-column, and we prove that the 3D-lifted method is strictly more accurate than full linearization.

\textbf{Step 3: Condition on time in 3D world space.} The linearized result is a joint Gaussian over (3D position $\times$ time). A key architectural insight is that the time conditioning should be performed in \emph{3D world space}, before the view-dependent perspective projection. Since the time component $z_0 = t$ is purely linear and view-independent, we can condition on $t = t_0$ to obtain a time-adjusted 3D mean and 3D covariance:
\[
V_{3\mathrm{D}}(t_0) = V_k + \vec{c}_{\mathrm{world}}\,\Sigma_{tt}^{-1}(t_0 - v_0), \qquad \Sigma_{3\mathrm{D}}(t_0) = \Sigma_{3\mathrm{D}} - \vec{c}_{\mathrm{world}}\,\Sigma_{tt}^{-1}\,\vec{c}_{\mathrm{world}}^T,
\]
where $\vec{c}_{\mathrm{world}} \in \mathbb{R}^3$ is the spatial-temporal cross-covariance vector. The opacity is modulated by an unnormalized temporal weight $w_t = \exp(-(t_0 - v_0)^2/2\Sigma_{tt})$. These are then passed directly to a \textbf{standard, unmodified 3D Gaussian splatting rasterizer}, which handles the view matrix $[R\mid t]$ and perspective projection natively. This avoids any custom CUDA code and eliminates the linearization error in the mean shift.

\subsection*{Key results}

\begin{enumerate}
\item \textbf{Explicit orthogonal basis} for $E_{p,q}$: $e_1 = p+q$ (purely spatial), $e_2 = 1-pq$ (mixes space and time). Both have norm $\sqrt{2(1+p\cdot q)}$. See Proposition~\ref{prop:basis}.

\item \textbf{Jacobian factorization} $J = J_{\mathrm{persp}} \cdot J_{\mathrm{embed}}$ (spatial) plus $J_{\mathrm{time}}$ (temporal), requiring only $d = p+q$, $s = p\times q$, and $c = p\cdot q$. See Proposition~\ref{prop:fullJ}.

\item \textbf{Rendering equation} with correct temporal opacity (using unnormalized Gaussian weight to keep $\alpha_k^{\mathrm{eff}} \in [0, \alpha_k]$):
\[
C(y \mid t_0) = \sum_{k} c_k \cdot \underbrace{\alpha_k \cdot \exp\!\Big(\!-\tfrac{(t_0-v_0)^2}{2\Sigma_{tt}}\Big)}_{\text{time-modulated opacity}} \cdot p_k(y\mid t_0) \cdot \prod_{j<k}\!\big(1 - \alpha_j^{\mathrm{eff}}(t_0)\cdot p_j(y\mid t_0)\big).
\]

\item \textbf{3D-lifted implementation recipe} (Section~\ref{sec:recipe}): time conditioning is performed in \emph{world space} (not pixel space), producing a standard 3D mean + 3D covariance that can be fed directly into an \emph{unmodified} 3DGS rasterizer. No custom CUDA code required.

\item \textbf{All gradients} for backpropagation through $(p,q)$, $\Sigma_k$, $v_0$, and the opacity, including Riemannian projection onto $S^2$ for the Grassmannian coordinates.

\item \textbf{Camera-aware analysis} (Section~\ref{sec:camera_aware}): For a \emph{static camera} (Case A), the Jacobian is exact up to EWA linearization, and the 3D-lifted method is exactly equivalent to full linearization. For a \emph{dynamic camera} (Case B), we derive the camera motion correction $\mathbf{m}$ and prove that the 3D-lifted method is strictly more accurate, with error $O(\norm{\mathbf{m}}\sqrt{\sigma_{\beta\beta}})$ (camera speed $\times$ temporal Gaussian width).
\end{enumerate}

\bigskip
\noindent\textit{The remainder of this document provides the proofs and detailed derivations.}

\bigskip
\tableofcontents

\section{Setup and Conventions}

\subsection{The ambient space}
We identify $\mathbb{R}^4$ with the quaternions $\HH$. A quaternion $x = x_0 + x_1 \mathbf{i} + x_2 \mathbf{j} + x_3 \mathbf{k}$ has:
\begin{itemize}
\item \textbf{Real (time) component:} $x_0 \in \mathbb{R}$,
\item \textbf{Imaginary (spatial) component:} $(x_1, x_2, x_3) \in \mathbb{R}^3$.
\end{itemize}
We write $x = (x_0, X)$ with $X = (x_1, x_2, x_3)$. The imaginary quaternions $\mathrm{Im}(\HH) \cong \mathbb{R}^3$ represent 3D space at time $t=0$.

\subsection{The Grassmannian parameterization}
For unit imaginary quaternions $p,q \in S^2 \subset \mathrm{Im}(\HH)$ with $p \neq -q$, the canonical plane is:
\[
E_{p,q} = \{x \in \HH : px = xq\}.
\]
This is a 2-dimensional real subspace of $\HH \cong \mathbb{R}^4$. The assignment $(p,q) \mapsto E_{p,q}$ gives the diffeomorphism $\mathrm{Gr}^+(4,2) \cong S^2 \times S^2$.

\subsection{The observation space}
The observation space is $\mathbb{R}^2 \times \mathbb{R}$, where:
\begin{itemize}
\item $\mathbb{R}^2$ is pixel space (image coordinates $(u,v)$),
\item $\mathbb{R}$ is time (the frame index $t$).
\end{itemize}
A point $z \in E_{p,q}$ encodes both spatial and temporal information. The full projection is:
\[
P: E_{p,q} \to \mathbb{R}^2 \times \mathbb{R}, \qquad z \mapsto (u,v,t).
\]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{An Explicit Basis for $E_{p,q}$}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{prop}\label{prop:basis}
The following two quaternions form an orthogonal basis of $E_{p,q}$:
\[
e_1 = p + q, \qquad e_2 = 1 - pq.
\]
They satisfy $\norm{e_1}^2 = \norm{e_2}^2 = 2(1+c)$, where $c = p\cdot q$ is the Euclidean inner product of $p$ and $q$ viewed as vectors in $\mathbb{R}^3$.
\end{prop}

\begin{proof}
\textbf{$e_1 = p+q$ lies in $E_{p,q}$:} Since $p$ is a unit imaginary quaternion, $p^2 = -1$. Thus:
\[
p(p+q) = p^2 + pq = -1 + pq, \qquad (p+q)q = pq + q^2 = pq -1.
\]
These are equal. \checkmark

\textbf{$e_2 = 1 - pq$ lies in $E_{p,q}$:}
\[
p(1-pq) = p - p^2q = p + q, \qquad
(1-pq)q = q - pq^2 = q + p.
\]
These are equal. \checkmark

\textbf{Decomposition in $\mathbb{R}^4$ coordinates.} Since $p,q$ are purely imaginary:
\[
e_1 = (0,\; p+q).
\]
For $e_2$: the quaternion product $pq = -p\cdot q + p\times q = -c + s$, where $s = p\times q \in \mathbb{R}^3$. So:
\[
e_2 = 1 - pq = (1+c,\; -s).
\]

\textbf{Orthogonality:}
\[
\ip{e_1,e_2}_{\mathbb{R}^4} = 0\cdot(1+c) + (p+q)\cdot(-s) = -(p+q)\cdot(p\times q) = 0,
\]
since $p\times q \perp p$ and $p\times q \perp q$. \checkmark

\textbf{Norms:}
\begin{align*}
\norm{e_1}^2 &= \norm{p+q}^2 = 2 + 2c, \\
\norm{e_2}^2 &= (1+c)^2 + \norm{s}^2 = (1+c)^2 + 1 - c^2 = 2(1+c). \qedhere
\end{align*}
\end{proof}

\begin{defn}[Shorthands]\label{def:shorthands}
Throughout, we define:
\[
c = p\cdot q, \quad d = p+q, \quad s = p\times q, \quad r = \frac{1}{\sqrt{2(1+c)}}.
\]
The orthonormal basis vectors are then:
\begin{align}
\hat{e}_1 &= r\,(0,\; d) & &\text{--- purely spatial, zero time component,} \label{eq:e1} \\
\hat{e}_2 &= r\,(1+c,\; -s) & &\text{--- has time component } r(1+c) = \sqrt{\tfrac{1+c}{2}}. \label{eq:e2}
\end{align}
\end{defn}

\begin{remark}[Geometric meaning of the basis]\label{rem:geometric}
The basis has a clean geometric interpretation:
\begin{itemize}
\item $\hat{e}_1$ points along $d = p+q$ in $\mathbb{R}^3$ with no time component. Moving along $\hat{e}_1$ moves \emph{along the line in space at a fixed time}.
\item $\hat{e}_2$ has spatial part $-s = -(p\times q)$ (perpendicular to both $p$ and $q$) and time part $r(1+c)$. Moving along $\hat{e}_2$ moves the line \emph{to a different time instant} while also shifting it spatially.
\end{itemize}
This is exactly what the Grassmannian model encodes: the 2-plane $E_{p,q}$ captures a line moving through time.
\end{remark}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{The Full Projection Map}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{Local coordinates}

A point in the canonical bundle near the mean $v = (v_0, V) \in E_{p,q}$ is parameterized as:
\[
z(\alpha,\beta) = v + \alpha\,\hat{e}_1 + \beta\,\hat{e}_2.
\]
The four components of $z$ in $\mathbb{R}^4 = (\text{time}, \text{space})$ are:
\begin{align}
z_0(\alpha,\beta) &= v_0 + \beta\, r(1+c), \label{eq:z0}\\
z_{\mathrm{spatial}}(\alpha,\beta) &= V + \alpha\, r\, d - \beta\, r\, s. \label{eq:zspatial}
\end{align}
Note that $\alpha$ affects \emph{only} the spatial components (since $\hat{e}_1$ is purely spatial), while $\beta$ affects \emph{both} time and space.

\subsection{The projection (camera-at-origin form)}\label{sec:proj_origin}

\begin{remark}[Coordinate convention]
The projection below is written in the coordinate system of a camera at the origin, looking along the $z$-axis. This is the natural frame for deriving the Jacobian (Section~\ref{sec:jacobian}). For a camera at position $c$ with rotation $R$, one first transforms world coordinates via $X_{\mathrm{cam}} = R(X_{\mathrm{world}} - c)$; this is handled by the rasterizer. In Section~\ref{sec:camera_aware}, we extend the analysis to moving cameras and show how the 3D-lifted architecture (Section~\ref{sec:recipe}) correctly handles the general case.
\end{remark}

The full projection $P: E_{p,q} \to \mathbb{R}^2 \times \mathbb{R}$ maps $z$ to pixel coordinates and time:
\begin{equation}\label{eq:fullP}
P(z) = \begin{pmatrix} u \\ v \\ t \end{pmatrix} = \begin{pmatrix} \dfrac{f_x\, z_1}{z_3} + c_x \\[8pt] \dfrac{f_y\, z_2}{z_3} + c_y \\[8pt] z_0 \end{pmatrix},
\end{equation}
where $(z_1, z_2, z_3) = z_{\mathrm{spatial}}$ and $z_0$ is the time component. The first two rows are the standard perspective projection (nonlinear due to division by $z_3$). The third row is simply the identity on the time component (linear).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{The Jacobian: Main Result}\label{sec:jacobian}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{prop}[Full Jacobian]\label{prop:fullJ}
The Jacobian of $P$ at the mean $v = (v_0, V)$ with $V = (V_1, V_2, V_3)$, in local coordinates $(\alpha,\beta)$, is the $3 \times 2$ matrix:
\begin{equation}\label{eq:Jfull}
\boxed{J_{\mathrm{full}} = \begin{pmatrix} J_{\mathrm{persp}} \cdot J_{\mathrm{embed}} \\[6pt] J_{\mathrm{time}} \end{pmatrix} \in \mathbb{R}^{3\times 2}}
\end{equation}
where each factor is defined as follows.
\end{prop}

\subsection{The perspective block: $J_{\mathrm{persp}} \cdot J_{\mathrm{embed}}$}

This is the standard EWA splatting Jacobian applied to the spatial embedding.

\begin{itemize}
\item $J_{\mathrm{persp}}$ is the $2\times 3$ Jacobian of the perspective projection at $V$:
\begin{equation}\label{eq:Jpersp}
J_{\mathrm{persp}} = \frac{1}{V_3}\begin{pmatrix} f_x & 0 & -f_x \frac{V_1}{V_3} \\ 0 & f_y & -f_y \frac{V_2}{V_3} \end{pmatrix}.
\end{equation}

\item $J_{\mathrm{embed}}$ is the $3\times 2$ matrix whose columns are the spatial parts of $\hat{e}_1$ and $\hat{e}_2$:
\begin{equation}\label{eq:Jembed}
J_{\mathrm{embed}} = r \begin{pmatrix} d_1 & -s_1 \\ d_2 & -s_2 \\ d_3 & -s_3 \end{pmatrix} = \frac{1}{\sqrt{2(1+c)}}\begin{pmatrix} p_1+q_1 & -(p\times q)_1 \\ p_2+q_2 & -(p\times q)_2 \\ p_3+q_3 & -(p\times q)_3 \end{pmatrix}.
\end{equation}

\item Their product $J_{\mathrm{persp}} \cdot J_{\mathrm{embed}}$ is a $2 \times 2$ matrix giving the spatial-to-pixel Jacobian.
\end{itemize}

\subsection{The time block: $J_{\mathrm{time}}$}

From \eqref{eq:z0}, the time component is $t = z_0 = v_0 + \beta\, r(1+c)$. Therefore:
\begin{equation}\label{eq:Jtime}
J_{\mathrm{time}} = \frac{\partial t}{\partial(\alpha,\beta)} = \begin{pmatrix} 0 & r(1+c) \end{pmatrix} = \begin{pmatrix} 0 & \sqrt{\dfrac{1+c}{2}} \end{pmatrix}.
\end{equation}
The zero in the first column confirms that $\alpha$ (motion along the spatial direction $\hat{e}_1$) does not change the time. The second entry $\sqrt{(1+c)/2}$ is the rate at which $\beta$ (motion along $\hat{e}_2$) advances time.

\subsection{The assembled Jacobian}

Combining:
\begin{equation}\label{eq:Jassembled}
\boxed{J_{\mathrm{full}} = \begin{pmatrix} \dfrac{r}{V_3}\Bigl(f_x(d_1 - \frac{V_1}{V_3}d_3) \Bigr) & \dfrac{r}{V_3}\Bigl(-f_x(s_1 - \frac{V_1}{V_3}s_3)\Bigr) \\[10pt] \dfrac{r}{V_3}\Bigl(f_y(d_2 - \frac{V_2}{V_3}d_3) \Bigr) & \dfrac{r}{V_3}\Bigl(-f_y(s_2 - \frac{V_2}{V_3}s_3)\Bigr) \\[10pt] 0 & r(1+c) \end{pmatrix}}
\end{equation}
where $d = p+q$, $s = p \times q$, $c = p \cdot q$, $r = 1/\sqrt{2(1+c)}$, and $V = (V_1, V_2, V_3)$ is the spatial part of the Gaussian mean.

\begin{proof}[Proof of Proposition~\ref{prop:fullJ}]
From \eqref{eq:z0} and \eqref{eq:zspatial}, the composite map in local coordinates is:
\[
P(\alpha,\beta) = \begin{pmatrix} \pi(V + \alpha\,r\,d - \beta\,r\,s) \\ v_0 + \beta\,r(1+c) \end{pmatrix}
\]
where $\pi$ is the perspective projection. By the chain rule:
\[
\frac{\partial P}{\partial(\alpha,\beta)}\bigg|_{(0,0)} = \begin{pmatrix} \frac{\partial \pi}{\partial (X,Y,Z)}\big|_V \cdot \frac{\partial z_{\mathrm{spatial}}}{\partial(\alpha,\beta)}\big|_{(0,0)} \\[6pt] \frac{\partial z_0}{\partial(\alpha,\beta)}\big|_{(0,0)} \end{pmatrix} = \begin{pmatrix} J_{\mathrm{persp}} \cdot J_{\mathrm{embed}} \\ J_{\mathrm{time}} \end{pmatrix}. \qedhere
\]
\end{proof}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Camera-Aware Projection}\label{sec:camera_aware}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The Jacobian derived in Section~\ref{sec:jacobian} assumes a camera fixed at the origin. In practice, the camera moves through the scene. We now extend the analysis to a general time-dependent camera pose and show that the two physically relevant cases --- static camera with moving scene, and moving camera --- have cleanly separated mathematical structures.

\subsection{General setup}

Let $R(t) \in \mathrm{SO}(3)$ and $c(t) \in \mathbb{R}^3$ denote the camera rotation and position at time $t$. A world-space point $X \in \mathbb{R}^3$ is transformed to camera coordinates via:
\[
X_{\mathrm{cam}}(t) = R(t)\big(X - c(t)\big).
\]
The full projection of a point $z = (z_0, z_{\mathrm{spatial}}) \in E_{p,q}$ is:
\begin{equation}\label{eq:P_general}
P(z) = \Big(\pi\!\big(R(z_0)\cdot(z_{\mathrm{spatial}} - c(z_0))\big),\;\; z_0\Big),
\end{equation}
where $\pi$ is the perspective projection \eqref{eq:fullP} and $z_0 = v_0 + \beta\, r(1+c)$ is the time component. The critical observation is that $R$ and $c$ are evaluated at $z_0$, which \emph{depends on} $\beta$. This couples the spatial and temporal coordinates through the camera trajectory.

\subsection{Case A: Static camera, moving scene}\label{sec:static_cam}

If the camera is fixed, then $R(t) = R_0$ and $c(t) = c_0$ for all $t$, so $\dot{R}_0 = 0$ and $\dot{c}_0 = 0$. The projection simplifies to:
\begin{equation}\label{eq:P_static}
P(z) = \Big(\pi\!\big(R_0(z_{\mathrm{spatial}} - c_0)\big),\;\; z_0\Big).
\end{equation}
The spatial and temporal components are now \textbf{fully decoupled}: the spatial projection $\pi(R_0(\cdot - c_0))$ does not depend on $z_0$.

\begin{prop}[Jacobian with static camera]\label{prop:J_static}
For a static camera $(R_0, c_0)$, the full Jacobian in local $(\alpha,\beta)$ coordinates is:
\begin{equation}\label{eq:J_static}
\boxed{J_{\mathrm{full}}^{\mathrm{static}} = \begin{pmatrix} J_\pi \cdot R_0 \cdot J_{\mathrm{embed}} \\[4pt] J_{\mathrm{time}} \end{pmatrix} \in \mathbb{R}^{3\times 2}}
\end{equation}
where $J_\pi$ is the $2 \times 3$ perspective Jacobian evaluated at $X_{\mathrm{cam}}^0 = R_0(V_k - c_0)$:
\[
J_\pi = \frac{1}{X_{\mathrm{cam},3}^0}\begin{pmatrix} f_x & 0 & -f_x\, X_{\mathrm{cam},1}^0 / X_{\mathrm{cam},3}^0 \\ 0 & f_y & -f_y\, X_{\mathrm{cam},2}^0 / X_{\mathrm{cam},3}^0 \end{pmatrix},
\]
and $J_{\mathrm{embed}}$, $J_{\mathrm{time}}$ are as before \eqref{eq:Jembed}, \eqref{eq:Jtime}.
\end{prop}

\begin{proof}
Define $\phi(X) = \pi(R_0(X - c_0))$, a fixed (time-independent) map from world-space to pixel-space. By the chain rule:
\[
\frac{\partial \phi}{\partial X}\bigg|_{V_k} = J_\pi \cdot R_0.
\]
Since $z_{\mathrm{spatial}}(\alpha,\beta) = V_k + \alpha\,r\,d - \beta\,r\,s$ (linear in $(\alpha,\beta)$):
\[
\frac{\partial \phi(z_{\mathrm{spatial}})}{\partial(\alpha,\beta)}\bigg|_{(0,0)} = J_\pi \cdot R_0 \cdot J_{\mathrm{embed}}.
\]
The time row is unchanged since $z_0$ does not depend on the camera.
\end{proof}

\begin{remark}[Relation to Section~\ref{sec:jacobian}]
Equation \eqref{eq:J_static} is identical to the camera-at-origin Jacobian \eqref{eq:Jfull} with the replacement $J_{\mathrm{persp}} \to J_\pi \cdot R_0$. The rotation $R_0$ simply rotates the embedding directions $d$ and $s$ into camera coordinates. All subsequent formulas (projected covariance, conditioning, rendering equation) carry through with this substitution.
\end{remark}

\begin{remark}[Exact equivalence with 3D-lifted method]
For a static camera, the 3D-lifted method (Section~\ref{sec:recipe}) and the full linearization give \textbf{identical results}. This is because the spatial projection $\pi \circ R_0 \circ (\cdot - c_0)$ is independent of time, so conditioning on $t = t_0$ in world space and then projecting gives exactly the same result as projecting the joint (space $\times$ time) Gaussian and then conditioning in pixel space. There is no approximation beyond the standard EWA linearization of $\pi$.
\end{remark}

This is the relevant case for the initial proof of concept: a monocular video captured from a fixed or approximately fixed viewpoint.

\subsection{Case B: Dynamic camera}\label{sec:dynamic_cam}

Now let $R(t)$ and $c(t)$ be smooth functions of time, with derivatives $\dot{R}_0 = \dot{R}(v_0)$ and $\dot{c}_0 = \dot{c}(v_0)$ at the temporal center $v_0$ of the Gaussian.

\begin{defn}[Camera motion vector]\label{def:motion_vec}
Define the \textbf{camera motion vector} at the Gaussian center:
\begin{equation}\label{eq:m_vec}
\mathbf{m} = \dot{R}_0(V_k - c_0) - R_0\,\dot{c}_0 \;\in\; \mathbb{R}^3.
\end{equation}
This is the velocity of the Gaussian center's image in camera coordinates, caused solely by camera motion. It has two contributions:
\begin{itemize}
\item $\dot{R}_0(V_k - c_0)$: camera rotation sweeps the point across the camera frame,
\item $-R_0\,\dot{c}_0$: camera translation shifts all points in camera coordinates.
\end{itemize}
\end{defn}

\begin{prop}[Jacobian with dynamic camera]\label{prop:J_dynamic}
For a dynamic camera, the Jacobian in local $(\alpha,\beta)$ coordinates is:
\begin{equation}\label{eq:J_dynamic}
\boxed{J_{\mathrm{full}}^{\mathrm{dynamic}} = \begin{pmatrix} J_\pi \cdot J_{\mathrm{cam}} \\[4pt] J_{\mathrm{time}} \end{pmatrix} \in \mathbb{R}^{3\times 2}}
\end{equation}
where $J_\pi$ is the perspective Jacobian (as in Case A), and the \textbf{camera-aware embedding Jacobian} $J_{\mathrm{cam}} \in \mathbb{R}^{3\times 2}$ is:
\begin{equation}\label{eq:Jcam}
\boxed{J_{\mathrm{cam}} = \begin{pmatrix} R_0\, r\, d & -R_0\, r\, s + \mathbf{m}\, r(1+c) \end{pmatrix}  = R_0\, J_{\mathrm{embed}} + \begin{pmatrix} \mathbf{0} & \mathbf{m}\, r(1+c) \end{pmatrix}.}
\end{equation}
\end{prop}

\begin{proof}
Define $X_{\mathrm{cam}}(\alpha,\beta) = R(z_0(\alpha,\beta))\big(z_{\mathrm{spatial}}(\alpha,\beta) - c(z_0(\alpha,\beta))\big)$. By the product rule and chain rule:
\[
\frac{\partial X_{\mathrm{cam}}}{\partial \alpha}\bigg|_{(0,0)} = R_0 \cdot \frac{\partial z_{\mathrm{spatial}}}{\partial\alpha} + \underbrace{\Big[\dot{R}_0(V_k - c_0) - R_0\,\dot{c}_0\Big]}_{=\,\mathbf{m}} \cdot \underbrace{\frac{\partial z_0}{\partial\alpha}}_{=\,0} = R_0 \cdot r\,d.
\]
The camera motion term vanishes because $\hat{e}_1$ has no time component ($\partial z_0/\partial\alpha = 0$).

For $\beta$:
\[
\frac{\partial X_{\mathrm{cam}}}{\partial \beta}\bigg|_{(0,0)} = R_0 \cdot \underbrace{\frac{\partial z_{\mathrm{spatial}}}{\partial\beta}}_{=\,-r\,s} + \;\mathbf{m} \cdot \underbrace{\frac{\partial z_0}{\partial\beta}}_{=\,r(1+c)} = -R_0\,r\,s + \mathbf{m}\,r(1+c).
\]
Here the camera motion contributes because $\hat{e}_2$ has a time component: moving along $\hat{e}_2$ changes $z_0$, which changes the camera pose, which shifts the image.
\end{proof}

\begin{remark}[Structure of the correction]
The dynamic Jacobian $J_{\mathrm{cam}}$ equals the static Jacobian $R_0\,J_{\mathrm{embed}}$ plus a rank-1 correction in the second column only:
\[
J_{\mathrm{cam}} = R_0\, J_{\mathrm{embed}} + \begin{pmatrix} \mathbf{0} & \mathbf{m}\, r(1+c) \end{pmatrix}.
\]
The correction only affects the $\beta$-direction (time-varying), not the $\alpha$-direction (purely spatial). Physically: the purely spatial direction $\hat{e}_1$ is invisible to the camera motion (same time instant), while the time-varying direction $\hat{e}_2$ picks up an extra apparent velocity from the camera.
\end{remark}

\subsection{3D-lifted method vs.\ full linearization}\label{sec:lifted_vs_full}

For a dynamic camera, the 3D-lifted method (Section~\ref{sec:recipe}) and the full linearization (Proposition~\ref{prop:J_dynamic}) are \emph{not} identical. We now characterize the difference precisely.

\textbf{Ansatz A (3D-lifted, Section~\ref{sec:recipe}):}
\begin{enumerate}
\item Condition the 4D Gaussian (world-space $\times$ time) on $t = t_0$. This is \textbf{exact} --- no linearization.
\item Transform the conditioned 3D Gaussian to camera coordinates using the \emph{exact} pose $(R_{t_0}, c_{t_0})$. This is \textbf{exact} --- a linear coordinate change.
\item Apply the standard EWA splatting (perspective linearization). This introduces the \textbf{only approximation}.
\end{enumerate}

\textbf{Ansatz B (full linearization with Proposition~\ref{prop:J_dynamic}):}
\begin{enumerate}
\item Linearize the full projection $P$ including the time-dependent camera. This introduces \textbf{two approximations}: linearization of $\pi$ (as in EWA) \emph{and} linearization of $R(t), c(t)$ around $t = v_0$.
\item Obtain a 3D Gaussian in $(u,v,t)$-space.
\item Condition on $t = t_0$. This is exact on the already-approximated distribution.
\end{enumerate}

\subsubsection{Notation for the comparison}

Define the following quantities, all evaluated at the Gaussian center:
\begin{align}
\tau &= r(1+c) = \sqrt{(1+c)/2} & &\text{(time scaling factor),} \notag \\
S &= J_\pi\, R_0\, J_{\mathrm{embed}} \in \mathbb{R}^{2\times 2} & &\text{(static spatial-to-pixel Jacobian),} \notag \\
\boldsymbol{\eta} &= J_\pi\, \mathbf{m}\, \tau \in \mathbb{R}^{2} & &\text{(projected camera velocity $\times$ time scale),} \notag \\
\mathbf{v} &= S \begin{pmatrix} \sigma_{\alpha\beta} \\ \sigma_{\beta\beta} \end{pmatrix} \in \mathbb{R}^2 & &\text{(projected covariance coupling vector).} \notag
\end{align}
For a static camera, $\boldsymbol{\eta} = 0$ and all correction terms below vanish.

\subsubsection{Ansatz B: projected covariance blocks}

The spatial block of the full Jacobian from Proposition~\ref{prop:J_dynamic} can be written as:
\begin{equation}\label{eq:JsB}
J_s^{(B)} = J_\pi\, J_{\mathrm{cam}} = S + \boldsymbol{\eta}\, e_2^T,
\end{equation}
where $e_2 = (0,1)^T$. That is: the first column ($\alpha$-direction) is the same as the static case; the second column ($\beta$-direction) picks up the projected camera velocity $\boldsymbol{\eta}$. Physically: motion in $\hat{e}_1$ doesn't change the time, so the camera ``doesn't know'' about it; motion in $\hat{e}_2$ changes the time, and the camera moves.

\textbf{Spatial--spatial block} ($2 \times 2$). Expanding $(S + \boldsymbol{\eta}\,e_2^T)\,\Sigma_k\,(S + \boldsymbol{\eta}\,e_2^T)^T$:
\begin{equation}\label{eq:Sigma_uv_B}
\Sigma_{uv}^{(B)} = \underbrace{S\,\Sigma_k\, S^T + \sigma_k^2\, I_2}_{=\;\Sigma_{uv}^{(A,\mathrm{lin})}} + \boldsymbol{\eta}\,\mathbf{v}^T + \mathbf{v}\,\boldsymbol{\eta}^T + \sigma_{\beta\beta}\,\boldsymbol{\eta}\boldsymbol{\eta}^T.
\end{equation}
The three correction terms are all proportional to $\boldsymbol{\eta}$: the first two (symmetric) mix camera motion with internal covariance; the last is purely quadratic in camera motion, weighted by the temporal variance $\sigma_{\beta\beta}$.

\textbf{Spatial--temporal block} ($2 \times 1$). Using $J_{\mathrm{time}} = (0,\;\tau)$:
\begin{equation}\label{eq:Sigma_uvt_B}
\Sigma_{uv,t}^{(B)} = J_s^{(B)}\,\Sigma_k\, J_{\mathrm{time}}^T = \underbrace{\tau\,\mathbf{v}}_{=\;\Sigma_{uv,t}^{(A)}} + \tau\,\sigma_{\beta\beta}\,\boldsymbol{\eta}.
\end{equation}
The correction $\tau\sigma_{\beta\beta}\boldsymbol{\eta}$ represents additional space--time correlation induced by the camera motion: even for a static scene, a moving camera creates apparent pixel--time correlation.

\textbf{Temporal--temporal block} (scalar). This is camera-independent:
\begin{equation}\label{eq:Sigma_tt_B}
\Sigma_{tt}^{(B)} = \Sigma_{tt}^{(A)} = \tau^2\,\sigma_{\beta\beta} + \sigma_k^2.
\end{equation}

\subsubsection{Ansatz B: conditioning on $t = t_0$}

Let $\Delta t = t_0 - v_0$. The conditioned pixel-space covariance under Ansatz~B is:
\[
\Sigma_{uv|t_0}^{(B)} = \Sigma_{uv}^{(B)} - \Sigma_{uv,t}^{(B)}\,\Sigma_{tt}^{-1}\,(\Sigma_{uv,t}^{(B)})^T.
\]
Substituting \eqref{eq:Sigma_uv_B}--\eqref{eq:Sigma_uvt_B} and expanding $(\tau\mathbf{v} + \tau\sigma_{\beta\beta}\boldsymbol{\eta})(\tau\mathbf{v} + \tau\sigma_{\beta\beta}\boldsymbol{\eta})^T$:
\begin{align*}
\Sigma_{uv|t_0}^{(B)} &= S\Sigma_k S^T + \sigma_k^2 I_2 + \boldsymbol{\eta}\mathbf{v}^T + \mathbf{v}\boldsymbol{\eta}^T + \sigma_{\beta\beta}\boldsymbol{\eta}\boldsymbol{\eta}^T \\
&\quad - \frac{1}{\Sigma_{tt}}\big[\tau^2 \mathbf{v}\mathbf{v}^T + \tau^2\sigma_{\beta\beta}(\mathbf{v}\boldsymbol{\eta}^T + \boldsymbol{\eta}\mathbf{v}^T) + \tau^2\sigma_{\beta\beta}^2\boldsymbol{\eta}\boldsymbol{\eta}^T\big].
\end{align*}
Collecting the $\boldsymbol{\eta}$-free terms gives $\Sigma_{uv|t_0}^{(A,\mathrm{lin})} = S\Sigma_k S^T + \sigma_k^2 I_2 - \tau^2\Sigma_{tt}^{-1}\mathbf{v}\mathbf{v}^T$. Define the \textbf{damping factor}:
\begin{equation}\label{eq:lambda}
\lambda = 1 - \frac{\tau^2\sigma_{\beta\beta}}{\Sigma_{tt}} = \frac{\sigma_k^2}{\Sigma_{tt}} \;\in\; (0,1].
\end{equation}
Then the $\boldsymbol{\eta}$-dependent terms factor cleanly:

\begin{equation}\label{eq:cov_diff_explicit}
\boxed{\Sigma_{uv|t_0}^{(B)} = \Sigma_{uv|t_0}^{(A,\mathrm{lin})} + \lambda\,\big(\boldsymbol{\eta}\,\mathbf{v}^T + \mathbf{v}\,\boldsymbol{\eta}^T + \sigma_{\beta\beta}\,\boldsymbol{\eta}\boldsymbol{\eta}^T\big)}
\end{equation}

\begin{remark}[Meaning of $\lambda$]
The damping factor $\lambda = \sigma_k^2 / \Sigma_{tt}$ is the ratio of the isotropic blur variance to the total temporal variance. When the Gaussian is temporally broad ($\Sigma_{tt} \gg \sigma_k^2$), $\lambda \approx 0$ and the correction vanishes --- conditioning ``absorbs'' the camera motion effect. When the Gaussian is temporally sharp ($\Sigma_{tt} \approx \sigma_k^2$), $\lambda \approx 1$ and the full correction remains. Intuitively: a temporally narrow Gaussian barely ``sees'' the camera motion, so there is little to correct.
\end{remark}

\subsubsection{The comparison theorem}

\begin{theo}[Approximation error comparison]\label{thm:comparison}
Let Ansatz~A (3D-lifted) and Ansatz~B (full linearization) be as described above. Then:

\medskip
\noindent\textbf{(i) Covariance difference.} The conditioned pixel-space covariances differ by:
\begin{equation}\label{eq:cov_error}
\Sigma_{uv|t_0}^{(B)} - \Sigma_{uv|t_0}^{(A)} = \lambda\,\big(\boldsymbol{\eta}\,\mathbf{v}^T + \mathbf{v}\,\boldsymbol{\eta}^T + \sigma_{\beta\beta}\,\boldsymbol{\eta}\boldsymbol{\eta}^T\big) + O(\norm{\dot{R}_0}^2 \Delta t^2),
\end{equation}
where the first term is the camera-motion correction (present only in Ansatz~B), and the $O(\cdot)$ term accounts for the fact that Ansatz~B uses $R_0 = R(v_0)$ while Ansatz~A uses $R_{t_0} = R(t_0)$.

\medskip
\noindent\textbf{(ii) Mean difference.} The conditioned means differ by:
\begin{equation}\label{eq:mean_error}
\mu_{uv|t_0}^{(B)} - \mu_{uv|t_0}^{(A)} = \underbrace{\frac{\tau\,\sigma_{\beta\beta}}{\Sigma_{tt}}\,\boldsymbol{\eta}\,\Delta t}_{\text{camera motion in mean shift}} + \underbrace{\big[\pi(X_{\mathrm{cam}}^0) + J_\pi^0\,\Delta X - \pi(X_{\mathrm{cam}}^{t_0})\big]}_{\text{combined perspective + camera linearization error}},
\end{equation}
where $\Delta X = R_0(\vec{c}_{\mathrm{world}}\,\Sigma_{tt}^{-1}\Delta t)$ is the camera-space mean shift. Ansatz~A evaluates the exact nonlinear projection $\pi(R_{t_0}(V_{3\mathrm{D}}(t_0) - c_{t_0}))$; Ansatz~B linearizes both $\pi$ and $R(t)$.

\medskip
\noindent\textbf{(iii) Static camera.} If $\dot{R}_0 = 0$ and $\dot{c}_0 = 0$, then $\boldsymbol{\eta} = 0$ and both expressions reduce to zero: the two methods are \textbf{identical} (up to the common EWA approximation).

\medskip
\noindent\textbf{(iv) Error scale.} The dominant correction terms scale as:
\begin{equation}\label{eq:error_scale}
\norm{\Sigma_{uv|t_0}^{(B)} - \Sigma_{uv|t_0}^{(A)}} = O\!\left(\lambda\,\norm{\boldsymbol{\eta}}\,(\norm{\mathbf{v}} + \sigma_{\beta\beta}\norm{\boldsymbol{\eta}})\right),
\end{equation}
where $\norm{\boldsymbol{\eta}} = \norm{J_\pi \mathbf{m}}\,\tau$ is the projected camera speed (pixels/frame) times the time scaling factor, and $\sigma_{\beta\beta}$ controls the temporal extent. In particular, the error is small when:
\begin{equation}\label{eq:smallness}
\norm{J_\pi\,\mathbf{m}} \cdot \tau \sqrt{\sigma_{\beta\beta}} \ll 1,
\end{equation}
i.e., projected camera speed (pixels/frame) $\times$ temporal Gaussian width (frames) $\ll$ 1 pixel.
\end{theo}

\begin{proof}
\textbf{Part (i).} Equation \eqref{eq:cov_diff_explicit} gives $\Sigma_{uv|t_0}^{(B)}$ in terms of $\Sigma_{uv|t_0}^{(A,\mathrm{lin})}$. The latter uses the static camera pose $(R_0, c_0)$, while the true Ansatz~A covariance $\Sigma_{uv|t_0}^{(A)}$ uses the exact pose $(R_{t_0}, c_{t_0})$. Since $R_{t_0} = R_0 + \dot{R}_0\,\Delta t + O(\Delta t^2)$, we have:
\[
\Sigma_{uv|t_0}^{(A)} = J_\pi^{(t_0)}\, R_{t_0}\, \Sigma_{3\mathrm{D}}(t_0)\, R_{t_0}^T\, (J_\pi^{(t_0)})^T + \sigma_k^2 I_2.
\]
Expanding $R_{t_0} \approx R_0 + \dot{R}_0\Delta t$ and collecting terms:
\[
\Sigma_{uv|t_0}^{(A,\mathrm{lin})} - \Sigma_{uv|t_0}^{(A)} = O(\norm{\dot{R}_0}\,\norm{\Sigma_{3\mathrm{D}}(t_0)}\,\Delta t) + O(\norm{\dot{R}_0}^2\,\Delta t^2).
\]
Combined with \eqref{eq:cov_diff_explicit}, this gives \eqref{eq:cov_error}. Note that the $\boldsymbol{\eta}$-terms are first-order in camera motion and present \emph{only} in Ansatz~B.

\textbf{Part (ii).} In Ansatz~B, the conditioned mean is:
\[
\mu_{uv|t_0}^{(B)} = \pi(X_{\mathrm{cam}}^0) + \frac{\Sigma_{uv,t}^{(B)}}{\Sigma_{tt}}\,\Delta t = \pi(X_{\mathrm{cam}}^0) + \frac{\tau\mathbf{v} + \tau\sigma_{\beta\beta}\boldsymbol{\eta}}{\Sigma_{tt}}\,\Delta t.
\]
In Ansatz~A, the conditioned mean is:
\[
\mu_{uv|t_0}^{(A)} = \pi\!\Big(R_{t_0}\big(V_{3\mathrm{D}}(t_0) - c_{t_0}\big)\Big),
\]
where $V_{3\mathrm{D}}(t_0) = V_k + \vec{c}_{\mathrm{world}}\,\Sigma_{tt}^{-1}\,\Delta t$ and the projection $\pi$ is evaluated \emph{exactly} (no Taylor expansion). The difference has two sources: the extra $\boldsymbol{\eta}$-term in the Ansatz~B mean shift, and the linearization of $\pi \circ R_{t_0}(\cdot)$ that Ansatz~B performs but Ansatz~A does not.

\textbf{Parts (iii) and (iv).} Part (iii) follows from $\mathbf{m} = 0 \Rightarrow \boldsymbol{\eta} = 0$. For Part (iv), the dominant term in \eqref{eq:cov_error} is the bilinear term $\lambda(\boldsymbol{\eta}\mathbf{v}^T + \mathbf{v}\boldsymbol{\eta}^T)$ with norm $O(\lambda\norm{\boldsymbol{\eta}}\norm{\mathbf{v}})$. The quadratic term $\lambda\sigma_{\beta\beta}\boldsymbol{\eta}\boldsymbol{\eta}^T$ has norm $O(\lambda\sigma_{\beta\beta}\norm{\boldsymbol{\eta}}^2)$. Since $\norm{\boldsymbol{\eta}} = \norm{J_\pi\mathbf{m}}\tau$ and $\norm{\mathbf{v}} \leq \norm{S}\sqrt{\sigma_{\alpha\beta}^2 + \sigma_{\beta\beta}^2}$, the combined error is controlled by $\norm{\boldsymbol{\eta}}\sqrt{\sigma_{\beta\beta}}$, yielding \eqref{eq:smallness}.
\end{proof}

\begin{remark}[Practical implications]
Condition \eqref{eq:smallness} has a vivid geometric meaning: consider a Gaussian with temporal width $\sqrt{\sigma_{\beta\beta}} \approx 10$ frames, and a camera moving at $\norm{J_\pi\mathbf{m}} \approx 5$ pixels/frame. The product is $\tau \cdot 5 \cdot 10 \approx 50\tau$ pixels --- far from $\ll 1$. For such Gaussians, Ansatz~B would significantly misestimate the covariance, while Ansatz~A handles this case exactly.

In practice, Gaussians that violate \eqref{eq:smallness} should be split by the adaptive density control: a Gaussian that ``sees'' many different camera poses during its temporal extent is too coarse a representation and benefits from being subdivided into temporally narrower components.

\textbf{Summary of approximation hierarchy:}
\begin{center}
\begin{tabular}{lcc}
\hline
\textbf{Source of error} & \textbf{Ansatz A} & \textbf{Ansatz B} \\
\hline
Perspective linearization (EWA) & Yes & Yes \\
Camera trajectory linearization & \textbf{No} & Yes ($\propto \boldsymbol{\eta}$) \\
Mean shift linearization & \textbf{No} & Yes \\
\hline
\textbf{Total approximations} & \textbf{1} & \textbf{3} \\
\hline
\end{tabular}
\end{center}
Use Ansatz~A (3D-lifted). It is simpler, more accurate, and requires no custom CUDA code.
\end{remark}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{The Projected Covariance in $(u,v,t)$-Space}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{The full projected covariance}

Let $\Sigma_k \in \mathrm{Sym}^+_2$ be the $2\times 2$ covariance of a Gaussian in the $(\alpha,\beta)$ coordinates of $E_{p,q}$. The linearized projection gives a Gaussian in $(u,v,t)$-space with $3 \times 3$ covariance:
\begin{equation}\label{eq:Sigma_proj}
\Sigma_{\mathrm{proj}} = J_{\mathrm{full}}\;\Sigma_k\; J_{\mathrm{full}}^T + \sigma_k^2\, I_3.
\end{equation}
This is a rank-2 matrix (plus the isotropic blur $\sigma_k^2 I_3$), reflecting the fact that the Gaussian lives in a 2-dimensional plane.

Write $\Sigma_{\mathrm{proj}}$ in block form corresponding to the $(u,v)$ and $t$ components:
\begin{equation}\label{eq:blockform}
\Sigma_{\mathrm{proj}} = \begin{pmatrix} \Sigma_{uv} & \Sigma_{uv,t} \\ \Sigma_{t,uv} & \Sigma_{tt} \end{pmatrix}, \qquad \mu_{\mathrm{proj}} = \begin{pmatrix} \mu_{uv} \\ \mu_t \end{pmatrix} = \begin{pmatrix} \pi(V) \\ v_0 \end{pmatrix}.
\end{equation}

\subsection{Conditioning on time: rendering a single frame}\label{sec:conditioning}

To render frame $t_0$, we condition the 3D Gaussian on $t = t_0$. By the standard formula for Gaussian conditioning:

\begin{equation}\label{eq:conditioned_mean}
\boxed{\mu_{uv|t_0} = \mu_{uv} + \Sigma_{uv,t}\;\Sigma_{tt}^{-1}\;(t_0 - \mu_t)}
\end{equation}

\begin{equation}\label{eq:conditioned_cov}
\boxed{\Sigma_{uv|t_0} = \Sigma_{uv} - \Sigma_{uv,t}\;\Sigma_{tt}^{-1}\;\Sigma_{t,uv}}
\end{equation}

\begin{remark}[Physical interpretation]
These formulas have a clear meaning:
\begin{itemize}
\item \textbf{Mean shift} \eqref{eq:conditioned_mean}: The projected center of the Gaussian on the image \emph{moves} as a function of time. The shift is $\Sigma_{uv,t}\,\Sigma_{tt}^{-1}\,(t_0 - \mu_t)$, which is a linear function of $t_0$. This encodes the apparent motion of the splat on the image plane --- effectively a local velocity. If the splat is purely spatial ($\Sigma_{uv,t} = 0$), there is no motion.
\item \textbf{Covariance shrinkage} \eqref{eq:conditioned_cov}: Conditioning on a specific time ``removes'' the temporal uncertainty, tightening the splat. The correction term $\Sigma_{uv,t}\,\Sigma_{tt}^{-1}\,\Sigma_{t,uv}$ is always positive semidefinite, so $\Sigma_{uv|t_0} \preceq \Sigma_{uv}$. The more correlated space and time are (i.e., the faster the object moves), the larger the correction.
\end{itemize}
\end{remark}

\subsection{Explicit block formulas}

Let $J_s = J_{\mathrm{persp}} \cdot J_{\mathrm{embed}} \in \mathbb{R}^{2\times 2}$ (the spatial-to-pixel Jacobian) and $J_t = J_{\mathrm{time}} \in \mathbb{R}^{1\times 2}$ (the time Jacobian). Write $\Sigma_k = \begin{pmatrix} \sigma_{\alpha\alpha} & \sigma_{\alpha\beta} \\ \sigma_{\alpha\beta} & \sigma_{\beta\beta} \end{pmatrix}$. Then:
\begin{align}
\Sigma_{uv} &= J_s\,\Sigma_k\, J_s^T + \sigma_k^2 I_2, \label{eq:Sigma_uv}\\
\Sigma_{tt} &= J_t\,\Sigma_k\, J_t^T + \sigma_k^2 = r^2(1+c)^2\,\sigma_{\beta\beta} + \sigma_k^2, \label{eq:Sigma_tt}\\
\Sigma_{uv,t} &= J_s\,\Sigma_k\, J_t^T \in \mathbb{R}^{2\times 1}. \label{eq:Sigma_cross}
\end{align}
Since $J_t = (0,\; r(1+c))$, the cross-covariance simplifies to:
\begin{equation}\label{eq:cross_explicit}
\Sigma_{uv,t} = r(1+c)\; J_s \begin{pmatrix} \sigma_{\alpha\beta} \\ \sigma_{\beta\beta} \end{pmatrix}.
\end{equation}

\begin{remark}[When does $\Sigma_{uv,t}$ vanish?]
The space-time cross-covariance vanishes when $J_s \cdot (\sigma_{\alpha\beta}, \sigma_{\beta\beta})^T = 0$. For generic $J_s$ (invertible), this requires $\sigma_{\alpha\beta} = \sigma_{\beta\beta} = 0$, meaning the Gaussian has zero variance in the $\hat{e}_2$ direction --- it is a line segment at a single instant, with no temporal extent. In this degenerate case, the model reduces to a standard spatial Gaussian at a fixed time.
\end{remark}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{The Linearized Rendering Equation}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{From intractable integral to closed form}

The original paper writes:
\[
p_k(y) = \int_{E_{p,q}} \mathcal{N}(y;\; P(z),\; \sigma_k^2 I)\;\mathcal{N}(z;\; z_k,\; \Sigma_k)\; dz,
\]
where $y = (u,v,t) \in \mathbb{R}^2 \times \mathbb{R}$ lives in the full observation space (pixel $\times$ time). With the linearization $P(z) \approx P(z_k) + J_{\mathrm{full}}(z - z_k)$, the integral becomes a convolution of two Gaussians and evaluates to:
\begin{equation}\label{eq:rendering_3d}
p_k(y, t) \approx \mathcal{N}\!\left(\begin{pmatrix} y \\ t \end{pmatrix};\;\; \begin{pmatrix} \pi(V_k) \\ v_0 \end{pmatrix},\;\; \Sigma_{\mathrm{proj}}\right)
\end{equation}
where $\Sigma_{\mathrm{proj}} = J_{\mathrm{full}}\,\Sigma_k\,J_{\mathrm{full}}^T + \sigma_k^2 I_3$ as in \eqref{eq:Sigma_proj}. This is a 3D Gaussian --- a joint density over pixel position \emph{and} time.

\subsection{Joint density = conditional $\times$ marginal}\label{sec:joint_decomp}

To render a specific frame $t_0$, we need the joint density $p_k(y, t_0)$, not the conditional $p_k(y \mid t_0)$. The standard factorization of a joint Gaussian gives:
\begin{equation}\label{eq:joint_factorization}
\boxed{p_k(y, t_0) = \underbrace{p_k(y \mid t_0)}_{\text{spatial shape}} \;\times\; \underbrace{p_k(t_0)}_{\text{temporal weight}}}
\end{equation}
where:
\begin{itemize}
\item $p_k(y \mid t_0) = \mathcal{N}(y;\; \mu_{uv|t_0},\; \Sigma_{uv|t_0})$ is the conditioned 2D Gaussian from Section~\ref{sec:conditioning}, giving the spatial shape and position of the splat at time $t_0$;
\item $p_k(t_0) = \mathcal{N}(t_0;\; v_0,\; \Sigma_{tt})$ is the marginal temporal density, giving how strongly the Gaussian is ``present'' at time $t_0$.
\end{itemize}

\begin{remark}[Why the temporal factor is essential]
Without $p_k(t_0)$, a Gaussian centered at $t = 0$ would contribute with identical opacity at $t = 100$. The conditioning alone shifts the spatial mean and adjusts the covariance, but does not attenuate the total mass. Physically, a splat that exists only at frame~0 must fade out at distant frames --- the marginal $\mathcal{N}(t_0; v_0, \Sigma_{tt})$ provides exactly this Gaussian decay.
\end{remark}

\subsection{Effective opacity}\label{sec:eff_opacity}

The temporal marginal naturally absorbs into the opacity. Define the \textbf{time-modulated effective opacity}:
\begin{equation}\label{eq:eff_opacity}
\boxed{\alpha_k^{\mathrm{eff}}(t_0) = \alpha_k \cdot w_t, \qquad w_t = \exp\!\left(-\frac{(t_0 - v_0)^2}{2\,\Sigma_{tt}}\right)}
\end{equation}
where $\alpha_k \in [0,1]$ is the learned base opacity and $\Sigma_{tt} = r^2(1+c)^2\,\sigma_{\beta\beta} + \sigma_k^2$ from \eqref{eq:Sigma_tt}.

\begin{remark}[Why unnormalized]
Note that $w_t$ is the Gaussian kernel \emph{without} the normalization factor $1/\sqrt{2\pi\Sigma_{tt}}$. This is critical: the normalized density $\mathcal{N}(t_0; v_0, \Sigma_{tt})$ carries a prefactor that diverges as $\Sigma_{tt} \to 0$, which would cause $\alpha_k^{\mathrm{eff}}$ to exceed 1 and break the alpha compositing. By dropping the normalization, we ensure $w_t \in [0,1]$ and hence $\alpha_k^{\mathrm{eff}} \in [0, \alpha_k] \subset [0,1]$. This is consistent with standard 3DGS, which also evaluates spatial Gaussians without the normalization constant (the peak value is always 1). The missing constant is absorbed by the learned opacity $\alpha_k$ during training.
\end{remark}

The temporal width $\Sigma_{tt}$ controls over how many frames the Gaussian is visible:
\begin{itemize}
\item Large $\Sigma_{tt}$: the Gaussian persists across many frames (e.g., a static background element).
\item Small $\Sigma_{tt}$: the Gaussian is active only near frame $v_0$ (e.g., a transient event).
\end{itemize}

\subsection{The full rendering equation}

\textbf{To render frame $t_0$:} for each Gaussian $k$, compute the conditioned spatial distribution $p_k(y \mid t_0)$ and the effective opacity $\alpha_k^{\mathrm{eff}}(t_0)$, then alpha-composite front-to-back:

\begin{equation}\label{eq:rendering_frame}
\boxed{C(y \mid t_0) = \sum_{k \in \text{sorted}} c_k \cdot \alpha_k^{\mathrm{eff}}(t_0) \cdot p_k(y \mid t_0) \cdot \prod_{j < k}\!\Big(1 - \alpha_j^{\mathrm{eff}}(t_0) \cdot p_j(y \mid t_0)\Big)}
\end{equation}
where:
\begin{itemize}
\item $c_k = \ip{w_k,\, \mathrm{sh}(p_k)\,\mathrm{sh}(q_k)}$ is the view- and line-dependent color from spherical harmonics,
\item the product $\prod_{j<k}(\cdots)$ is the accumulated transmittance (standard alpha compositing),
\item the sum is over Gaussians sorted by depth at frame $t_0$.
\end{itemize}

\begin{remark}[Consistency with the original paper]
The original paper writes $p(y) = \sum \alpha_k\, p_k(y)$ with $y \in A \times \mathbb{R}$, which is the joint density evaluated at $(u,v,t_0)$. This is \emph{implicitly} correct: since $p_k(y)$ is the full joint density, the temporal attenuation is baked in. Our decomposition \eqref{eq:joint_factorization} simply makes explicit what was implicit, separating it into a form suitable for implementation: a 2D Gaussian (for the rasterizer) times a scalar temporal weight (for the opacity). The original paper also omits the transmittance product, which is needed for correct occlusion handling.
\end{remark}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Gradients for Backpropagation}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{Gradient w.r.t.\ $\Sigma_k$}

Since $\Sigma_{\mathrm{proj}}$ is linear in $\Sigma_k$, and the conditioning formulas \eqref{eq:conditioned_mean}--\eqref{eq:conditioned_cov} are differentiable functions of $\Sigma_{\mathrm{proj}}$, the chain rule gives:
\[
\frac{\partial \mathcal{L}}{\partial \Sigma_k} = J_{\mathrm{full}}^T\; \frac{\partial \mathcal{L}}{\partial \Sigma_{\mathrm{proj}}}\; J_{\mathrm{full}}.
\]
The inner gradient $\partial\mathcal{L}/\partial\Sigma_{\mathrm{proj}}$ can be obtained from $\partial\mathcal{L}/\partial\Sigma_{uv|t_0}$ and $\partial\mathcal{L}/\partial\mu_{uv|t_0}$ by differentiating through the conditioning equations.

\subsection{Gradient w.r.t.\ $V_k$ (spatial mean)}

$V_k$ enters through two paths:
\begin{enumerate}
\item Through $\pi(V_k) = \mu_{uv}$ (the projected mean). This gives the standard 3DGS gradient.
\item Through $J_{\mathrm{persp}}$ (which depends on $V_3$). This also exists in standard 3DGS.
\end{enumerate}
Both are already implemented in existing rasterizers. No new derivation needed.

\subsection{Gradient w.r.t.\ $v_0$ (temporal mean)}

$v_0$ enters through \emph{two} paths:
\begin{enumerate}
\item Through the conditioned 3D spatial mean $V_{3\mathrm{D}}(t_0)$ (Section~\ref{sec:3dlift}).
\item Through the effective opacity $\alpha_k^{\mathrm{eff}}(t_0) = \alpha_k \cdot w_t$ via \eqref{eq:eff_opacity}.
\end{enumerate}
The total gradient is:
\[
\frac{\partial \mathcal{L}}{\partial v_0} = \underbrace{-\frac{\partial \mathcal{L}}{\partial V_{3\mathrm{D}}} \cdot \vec{c}_{\mathrm{world}}\,\Sigma_{tt}^{-1}}_{\text{from 3D spatial shift}} \;+\; \underbrace{\frac{\partial \mathcal{L}}{\partial \alpha_k^{\mathrm{eff}}} \cdot \alpha_k^{\mathrm{eff}}(t_0) \cdot \frac{t_0 - v_0}{\Sigma_{tt}}}_{\text{from temporal attenuation}}.
\]
The second term drives $v_0$ toward $t_0$ if the Gaussian is useful at frame $t_0$: it pulls the temporal center to times where the Gaussian is needed.

\subsection{Gradient w.r.t.\ $(p,q)$}

The dependence on $(p,q)$ enters through:
\begin{itemize}
\item $J_{\mathrm{embed}}$ (via $d = p+q$, $s = p\times q$, $r = 1/\sqrt{2(1+c)}$),
\item $J_{\mathrm{time}}$ (via $r(1+c) = \sqrt{(1+c)/2}$),
\item $V_k$ (since the mean $v$ lies in $E_{p,q}$; moving $(p,q)$ moves the plane).
\end{itemize}

The elementary derivatives are:
\begin{align}
\frac{\partial d}{\partial p} &= I_{3\times 3}, & \frac{\partial d}{\partial q} &= I_{3\times 3}, \\[4pt]
\frac{\partial s}{\partial p} &= -[q]_\times, & \frac{\partial s}{\partial q} &= [p]_\times, \\[4pt]
\frac{\partial r}{\partial p} &= -\frac{r\, q}{2(1+c)}, & \frac{\partial r}{\partial q} &= -\frac{r\, p}{2(1+c)}, \\[4pt]
\frac{\partial [r(1+c)]}{\partial p} &= \frac{q}{2\sqrt{2(1+c)}}, & \frac{\partial [r(1+c)]}{\partial q} &= \frac{p}{2\sqrt{2(1+c)}},
\end{align}
where $[a]_\times$ is the $3\times 3$ skew-symmetric cross-product matrix.

After computing the Euclidean gradient $\nabla_p \mathcal{L} \in \mathbb{R}^3$, project onto $T_p S^2$:
\[
\mathrm{grad}_{S^2}\mathcal{L}\big|_p = \nabla_p \mathcal{L} - (p \cdot \nabla_p \mathcal{L})\, p,
\]
and update via retraction (e.g., normalize after an Adam step).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Summary: Implementation Recipe}\label{sec:recipe}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\subsection{3D-lifted time conditioning}\label{sec:3dlift}

A crucial architectural insight is that the time conditioning should be performed in \textbf{3D world space}, not in 2D pixel space. This is possible because the time component $z_0 = t$ is purely linear and view-independent. By conditioning in world space, we:
\begin{itemize}
\item avoid recomputing view-dependent quantities ($J_{\mathrm{persp}}$, $\Sigma_{uv,t}$) per frame,
\item pass the mean shift through the \emph{exact} nonlinear projection $\pi(V_{3\mathrm{D}}(t_0))$, rather than the linearized approximation $\pi(V_k) + J_{\mathrm{persp}}\,\Delta V$,
\item produce a standard (3D position, 3D covariance, opacity) triple that can be fed into a completely unmodified 3DGS rasterizer.
\end{itemize}

Define the \textbf{3D spatial-temporal cross-covariance vector}:
\begin{equation}\label{eq:c_world}
\vec{c}_{\mathrm{world}} = r(1+c)\; J_{\mathrm{embed}} \begin{pmatrix} \sigma_{\alpha\beta} \\ \sigma_{\beta\beta} \end{pmatrix} \in \mathbb{R}^3.
\end{equation}
This is the view-independent analog of $\Sigma_{uv,t}$ from \eqref{eq:cross_explicit}: it encodes how the 3D position shifts with time.

The conditioned 3D mean and 3D covariance for frame $t_0$ are:
\begin{equation}\label{eq:V3d_conditioned}
\boxed{V_{3\mathrm{D}}(t_0) = V_k + \vec{c}_{\mathrm{world}}\;\Sigma_{tt}^{-1}\;(t_0 - v_0)}
\end{equation}
\begin{equation}\label{eq:Sigma3d_conditioned}
\boxed{\Sigma_{3\mathrm{D}}(t_0) = \Sigma_{3\mathrm{D}} - \vec{c}_{\mathrm{world}}\;\Sigma_{tt}^{-1}\;\vec{c}_{\mathrm{world}}^T}
\end{equation}
where $\Sigma_{3\mathrm{D}} = J_{\mathrm{embed}}\,\Sigma_k\,J_{\mathrm{embed}}^T$ and $\Sigma_{tt}$ are as before.

\begin{remark}[Rank structure after conditioning]\label{rem:rank}
Before conditioning, $\Sigma_{3\mathrm{D}}$ has rank 2 (the Gaussian lives in a 2-plane). After conditioning on time, $\Sigma_{3\mathrm{D}}(t_0)$ has \textbf{rank 1}: a moving line sliced at a single time instant is a 1D line segment, which projects to a 1D structure on the image. It is only the rasterizer's pixel-space low-pass filter ($\sigma_k^2 I$) that fattens this line into a full-rank 2D splat. This is a geometrically correct feature of the model.
\end{remark}

\subsection{The algorithm}

\noindent\textbf{Input per Gaussian:} $(p,q) \in S^2 \times S^2$, $(\alpha_0, \beta_0) \in \mathbb{R}^2$ (mean in $E_{p,q}$), $\Sigma_k \in \mathrm{Sym}^+_2$, opacity $o_k$, SH coefficients.

\medskip

\noindent\textbf{Preprocessing (view-independent, computed once per Gaussian):}
\begin{enumerate}
\item Compute $d = p+q$, $s = p \times q$, $c = p \cdot q$, $r = 1/\sqrt{2(1+c)}$.
\item Compute spatial mean: $V_k = \alpha_0\, r\, d - \beta_0\, r\, s$. Temporal mean: $v_0 = \beta_0\, r(1+c)$.
\item Compute $J_{\mathrm{embed}}$ from \eqref{eq:Jembed}: a $3\times 2$ matrix from $d, s, r$.
\item Compute $\Sigma_{3\mathrm{D}} = J_{\mathrm{embed}}\,\Sigma_k\, J_{\mathrm{embed}}^T$ (a rank-2 $3\times 3$ matrix).
\item Compute $\Sigma_{tt} = r^2(1+c)^2\,\sigma_{\beta\beta} + \sigma_k^2$ and $\vec{c}_{\mathrm{world}}$ from \eqref{eq:c_world}.
\end{enumerate}

\noindent\textbf{Per-frame conditioning (for frame $t_0$, in world space):}
\begin{enumerate}\setcounter{enumi}{5}
\item Temporal weight: $w_t = \exp(-(t_0 - v_0)^2 / 2\Sigma_{tt})$, effective opacity: $\alpha_k^{\mathrm{eff}} = o_k \cdot w_t$.
\item Conditioned 3D mean: $V_{3\mathrm{D}}(t_0) = V_k + \vec{c}_{\mathrm{world}}\,\Sigma_{tt}^{-1}(t_0 - v_0)$.
\item Conditioned 3D covariance: $\Sigma_{3\mathrm{D}}(t_0) = \Sigma_{3\mathrm{D}} - \vec{c}_{\mathrm{world}}\,\Sigma_{tt}^{-1}\,\vec{c}_{\mathrm{world}}^T$.
\end{enumerate}

\noindent\textbf{Rendering (standard 3DGS rasterizer, unmodified):}
\begin{enumerate}\setcounter{enumi}{8}
\item Feed $(V_{3\mathrm{D}}(t_0),\; \Sigma_{3\mathrm{D}}(t_0),\; \alpha_k^{\mathrm{eff}},\; \text{color})$ into the standard \textbf{3D} rasterizer (e.g., \texttt{diff-gaussian-rasterization} with \texttt{cov3D\_precomp}).
\end{enumerate}
The rasterizer handles the view matrix $[R\mid t]$, perspective Jacobian $J_{\mathrm{persp}}$, tile assignment, depth sorting, and alpha compositing \emph{natively}. No custom CUDA code is needed.

\medskip

\noindent\textbf{Culling optimization:} Gaussians with $\alpha_k^{\mathrm{eff}}(t_0) < \varepsilon$ (e.g., $\varepsilon = 1/255$) can be skipped entirely for frame $t_0$. Since $w_t$ decays as a Gaussian in $|t_0 - v_0|$, this provides a natural temporal culling radius of $\sim 2\sqrt{\Sigma_{tt}}$ frames around $v_0$.

\medskip

\noindent\textbf{Gradient computation:} Steps 1--8 are standard PyTorch operations (quaternion arithmetic, matrix multiplies, scalar functions). PyTorch autograd will automatically compute gradients for $(p,q)$, $\Sigma_k$, $v_0$, $\alpha_0$, $\beta_0$, and $o_k$ through these operations. The rasterizer provides $\partial\mathcal{L}/\partial V_{3\mathrm{D}}$, $\partial\mathcal{L}/\partial\Sigma_{3\mathrm{D}}$, and $\partial\mathcal{L}/\partial\alpha_k^{\mathrm{eff}}$ via its built-in backward pass. The only non-standard gradient step is the Riemannian projection onto $S^2$ for $(p,q)$ updates (Section 7.4).

\medskip

\noindent\textbf{Parameter count per Gaussian:}
\begin{center}
\begin{tabular}{lcc}
\hline
\textbf{Parameter} & \textbf{Symbol} & \textbf{DOF} \\
\hline
Line (Grassmannian) & $p \in S^2$ & 2 \\
Line (Grassmannian) & $q \in S^2$ & 2 \\
Mean in canonical plane & $(\alpha_0, \beta_0) \in \mathbb{R}^2$ & 2 \\
Covariance in canonical plane & $\Sigma_k \in \mathrm{Sym}^+_2$ & 3 \\
\hline
\textbf{Geometry subtotal} & & \textbf{9} \\
\hline
Opacity & $o_k \in [0,1]$ & 1 \\
SH coefficients (degree $\ell$) & $w_{p,q}$ & $(\ell+1)^2$ per channel \\
\hline
\end{tabular}
\end{center}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Comparison with Standard 3DGS}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{center}
\begin{tabular}{lcc}
\hline
& \textbf{Standard 3DGS} & \textbf{Grassmannian model} \\
\hline
Primitive & Point in $\mathbb{R}^3$ & Line in $\mathbb{R}^3$ via $(p,q) \in S^2 \times S^2$ \\
Ambient space & $\mathbb{R}^3$ & $\mathbb{R}^4$ (space $\times$ time) \\
Mean & $\mu \in \mathbb{R}^3$ (3 DOF) & $v \in E_{p,q}$ (2 DOF) \\
Covariance & $\Sigma \in \mathrm{Sym}^+_3$ (6 DOF) & $\Sigma \in \mathrm{Sym}^+_2$ (3 DOF) \\
Geometry DOF & 9 & 9 \\
Temporal info & None (static) & Encoded in $\hat{e}_2$ direction \\
Rendering & Direct rasterization & 3D conditioning on $t_0$, then standard rasterization \\
New CUDA code & --- & \textbf{None} (PyTorch preprocessing only) \\
Rasterizer & Standard 3D & \textbf{Same, unmodified} (via \texttt{cov3D\_precomp}) \\
\hline
\end{tabular}
\end{center}

\begin{remark}[Rank structure is physically meaningful]
The matrix $\Sigma_{3\mathrm{D}} = J_{\mathrm{embed}}\,\Sigma_k\, J_{\mathrm{embed}}^T$ has rank at most 2 (the Gaussian lives in a 2-plane, not full 3D). After conditioning on $t = t_0$, the rank drops further: $\Sigma_{3\mathrm{D}}(t_0)$ has \textbf{rank 1}. This is geometrically exact --- a moving line, sliced at a single time instant, is a 1D line segment. Projected onto the 2D image plane, it remains a 1D structure. It is only the rasterizer's pixel-space low-pass filter ($+\sigma_k^2 I_2$ in screen space) that fattens this mathematical line segment into a full-rank 2D splat suitable for rendering. This rank progression ($2 \to 1 \to 2$ via the blur) is a satisfying confirmation that the model correctly captures the geometry of a line moving through time.
\end{remark}

\subsection*{Acknowledgments}
We thank the reviewer who identified the view-matrix bug (Point 1), the rank-deficiency correction (Point 2), and the opacity normalization issue (Point 3) in v4 of this note. All three corrections are incorporated in v5.

\end{document}