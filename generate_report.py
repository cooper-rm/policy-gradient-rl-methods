import subprocess
import os

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
TEX_FILE = os.path.join(REPORT_DIR, "Cooper_Morgan_Lab6.tex")
PDF_FILE = os.path.join(REPORT_DIR, "Cooper_Morgan_Lab6.pdf")

FIGURE_1 = "figures/figure1.png"
FIGURE_2 = "figures/figure2.png"
FIGURE_3 = "figures/figure3.png"
FIGURE_4 = "figures/figure4.png"
FIGURE_5 = "figures/policy_heatmap.png"

tex_content = r"""
\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{parskip}
\usepackage{float}

\titleformat{\section}{\large\bfseries}{}{0em}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}

\title{Lab 6: PyTorch and Actor-Critic Methods}
\author{Morgan Cooper \\ MSDS 684 --- Reinforcement Learning}
\date{\today}


\begin{document}
\maketitle
\newpage

\section{Section 1: Project Overview}

This lab moves from value-based methods to policy-based methods. Instead of learning
Q-values and deriving a policy from them like in Lab 5, the actor-critic algorithm learns
the policy directly with a neural network. A second neural network, the critic, learns the
state value $V(s)$ and tells the actor how good a state is. The actor uses this signal as
an advantage to update itself with less variance than REINFORCE alone. PyTorch is used
here for the first time so that gradients are computed automatically instead of by hand.

Pendulum-v1 has a continuous action space, so a softmax policy like in CartPole will not
work. Instead the actor outputs the mean of a Gaussian distribution and learns the standard
deviation as a parameter. Actions are sampled from this distribution and clipped to the
action range before stepping the environment. The critic uses the same kind of value
function approximator as Lab 5, but now it is a neural network instead of tile coding, and
it is updated with TD(0) bootstrapping every step.

\textbf{Pendulum-v1 (Gymnasium):}

\begin{itemize}
  \item State space: Box([-1, -1, -8], [1, 1, 8]) --- $\cos\theta$, $\sin\theta$, and angular velocity
  \item Action space: 1 continuous action --- torque between $-2$ and $+2$
  \item Rewards: $-(\theta^2 + 0.1\,\dot\theta^2 + 0.001\,a^2)$ per step, best near $0$ when upright and still
  \item Terminal condition: 200 step time limit, no early termination
\end{itemize}

I hypothesize that training will be unstable at first because actor-critic with TD(0) is
known to have high variance. From earlier labs I expect to need several debugging cycles
to fix issues like the actor mean saturating outside the action range or the standard
deviation drifting too high. Once those are fixed, I expect the agent to learn to swing
the pendulum upright and hold it there. Running 30 random seeds should show how reliable
the method is across different initializations.

\newpage
\section{Section 2: Deliverables}

\subsection{GitHub Repository}
\begin{verbatim}
GitHub Repository: https://github.com/cooper-rm/policy-gradient-rl-methods
\end{verbatim}

\subsection{Implementation Summary}

I implemented online actor-critic with TD(0) on Pendulum-v1 using PyTorch. The actor is
a neural network with two hidden layers of 64 tanh units. It outputs the mean of a
Gaussian policy. The standard deviation is a separate learned parameter clamped so sigma
stays between 0.13 and 0.50. The critic uses the same hidden layers but outputs a single
state value $V(s)$. Both networks use Adam with the actor at 3e-4 and the critic at 1e-3.
I set gamma=0.99, entropy\_coef=0.01, and grad\_clip=1.0. Furthermore, I added several
fixes during debugging including a tanh squash on the actor mean, a best-checkpoint
restore, and reward shaping that encourages staying upright with small actions. I ran
30 independent seeds with 1200 episodes each.

\subsection{Key Results \& Analysis}

The 30-seed sweep gave a mean final return of $-1210$ with a standard deviation of 363.
Out of 30 seeds, only 4 reached a return better than $-500$, and most plateaued around
$-1400$. This shows the method works but only some of the time, which matches what
Sutton and Barto describe in \S13.5 about the high variance of one-step actor-critic.
The wide 95\% CI in Figure~\ref{fig:fig1} mostly comes from this split between
successful and failed seeds.

The successful seeds learned a static balance rather than active stabilization. In
Figure~\ref{fig:fig3}, episodes 0 and 1 settle at $\theta \approx -0.25$ and
$\dot\theta \approx 0$ with a constant torque around $+0.6$. This makes physical sense
because at that angle gravity needs $+0.6$ torque to cancel out, so the agent learned to
provide exactly that. Interestingly, in an earlier experiment the agent did oscillate
around $\theta = 0$ with small corrective torques, which was the active stabilization I
originally expected. After I added reward shaping that rewards being upright with small
actions, the agent shifted to the simpler static-tilt solution. The policy heatmap in
Figure~\ref{fig:fig5} shows why this works. The agent outputs positive torque almost
everywhere with a narrow negative-torque region near upright, so once the pendulum
balances it sits inside the positive region and stays there.

A few hyperparameters made a big difference. The actor mean needed a tanh squash to
keep it inside the action range. Without it, $\mu(s)$ drifted outside $\pm 2$ and every
action ended up clipped to either $-2$ or $+2$. The standard deviation also needed a
clamp, because the actor gradient kept pushing $\sigma$ upward no matter what, which is
why the entropy curve in Figure~\ref{fig:fig2} is flat at the clamp value. Gradient
clipping at 1.0 was enough to stop late-training collapse, but at 0.5 the actor barely
learned at all. I also tried a reward shaping term that only penalized actions, but this
made the agent settle at a tilted equilibrium with a large constant torque, since holding
constant torque is cheaper than constantly correcting. After that I switched to
penalizing tilt and rewarding being upright too. The TD-error spikes in
Figure~\ref{fig:fig4} are a good example of how unstable training is, since they show
the critic having a hard time keeping up with how fast the actor is changing. Overall,
online TD(0) actor-critic was more fragile than I expected, and small things like the
action range or $\sigma$ growing unchecked can completely break training. This is
probably why methods like PPO and SAC are more popular for this kind of problem.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{""" + FIGURE_1 + r"""}
\caption{Mean return $\pm$ 95\% CI across 30 independent seeds, smoothed with a
25-episode moving window. Each seed trains its own actor and critic from scratch with
its own random initialization and environment reset seeds.}
\label{fig:fig1}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{""" + FIGURE_2 + r"""}
\caption{Mean policy entropy across training for a single seed (1200 episodes).
Entropy is computed from $\sigma$ of the Gaussian policy at each step and averaged
per episode. Higher entropy means more exploration.}
\label{fig:fig2}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{""" + FIGURE_3 + r"""}
\caption{Three sample rollouts from the trained single-seed policy with reset seeds
123, 124, and 125. Top: pendulum angle $\theta$ in radians. Middle: angular velocity
$\dot\theta$. Bottom: torque applied by the policy each step.}
\label{fig:fig3}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{""" + FIGURE_4 + r"""}
\caption{Mean $|\delta|$ per episode, where $\delta = r + \gamma V(s') - V(s)$ is the
TD error used to update both networks. The critic minimizes $\delta^2$ directly, and the
actor uses $\delta$ as the advantage in its policy gradient update.}
\label{fig:fig4}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{""" + FIGURE_5 + r"""}
\caption{Mean action $\mu(s)$ of the trained policy across a 2-D slice of the state
space, with $\theta$ on the x-axis and $\dot\theta$ on the y-axis. The 3-D state is
reconstructed as $(\cos\theta, \sin\theta, \dot\theta)$ before passing through the actor
network. Color shows the torque the policy would apply.}
\label{fig:fig5}
\end{figure}

\section{Section 3: AI Use Reflection}

\subsection{Initial Interaction}

Like previous labs, I started by talking through the week's concepts in ChatGPT. I had
it review policy gradients and actor-critic methods from Sutton and Barto Chapter 13,
then quiz me on the difference between value-based and policy-based methods. Once I felt
comfortable with the concepts, I switched to Claude Code in Visual Studio Code. I asked
Claude to review the lab directions and build a notebook template with markdown cells
for each section before writing any code.

\subsection{Iteration Cycle}

\textbf{Iteration 1: Actor mean saturated outside the action range}

After the first training run, I looked at the policy heatmap and saw the colorbar running
$\pm 15$ even though Pendulum's action space is $\pm 2$. The actor was outputting $\mu$
values way outside the legal range, and torch.clamp was doing all the work. I worked with
Claude to add a tanh squash on the mean output so it stays inside the action range. After
this fix the heatmap looked clean and training actually started learning a real policy.

\textbf{Iteration 2: Standard deviation grew without bound}

The next training run had log\_std drift up to about $+2.0$, which means $\sigma \approx 7.85$.
This was way bigger than the action range, so sampled actions were basically noise that got
clipped. Claude explained that the actor gradient consistently pushes $\sigma$ upward when
advantages are noisy, so I added a clamp on log\_std to keep $\sigma$ in a reasonable range.
This kept training stable for the rest of the experiments.

\textbf{Iteration 3: Reward shaping pushed the agent to a lazy solution}

I added reward shaping to encourage smaller actions when the pendulum was balanced. This
made the agent settle at a tilted equilibrium with a constant torque around $-1.25$ instead
of actively stabilizing at $\theta = 0$. After working through the math with Claude, I
realized that a slight tilt with constant torque is cheaper than constantly correcting, so
the agent took the easy way out. I switched the shaping to penalize tilt as well as actions,
which moved the agent closer to upright.

\subsection{Critical Evaluation}

Compared to previous labs, Claude required more iterations during this lab because
actor-critic has many ways to fail. Most issues came down to PyTorch interactions like the
action range mismatch and $\sigma$ growing without bound, which were not obvious until I
looked at the figures. I had to check each fix against Sutton and Barto, but Claude was
helpful for explaining the gradient behavior at each step.

\subsection{Learning Reflection}

The main lesson from this lab is that policy gradient methods are much more sensitive to
setup than value-based methods. With SARSA or Q-learning, even rough hyperparameters give
reasonable results. With actor-critic, things like the action range, the $\sigma$ clamp,
and reward shaping completely change what the agent learns. Visualizations were critical
for catching these differences since the failure modes were not obvious from the loss curves
alone.

\section{Section 4: Speaker Notes}

\begin{itemize}
  \item \textbf{Problem:} Train an actor-critic agent to swing up and balance Pendulum-v1.
  \item \textbf{Method:} Online actor-critic with TD(0) using a Gaussian policy and value network, both in PyTorch.
  \item \textbf{Design choice:} Use a state-independent learned $\sigma$ with a clamp to keep it from blowing up.
  \item \textbf{Key result:} Mean return of $-1210$ across 30 seeds (std 363); best seed reached $-264$; only 4 of 30 seeds succeeded.
  \item \textbf{Insight:} Reward shaping changes which equilibrium the agent settles at, sometimes for the worse.
  \item \textbf{Challenge:} The actor mean kept saturating outside the action range until I added a tanh squash.
  \item \textbf{Connection:} Methods like PPO and SAC build on this base to fix the instabilities seen in simple actor-critic.
\end{itemize}

\section{References}

\begin{enumerate}
  \item Sutton, R. S., \& Barto, A. G. (2018). \textit{Reinforcement learning: An introduction} (2nd ed.). MIT Press.
  \item Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. \textit{NeurIPS}.
  \item Anthropic. (2025). Claude Code [Large language model CLI tool]. \texttt{https://claude.ai}
  \item OpenAI. (2025). ChatGPT [Large language model]. \texttt{https://chat.openai.com}
\end{enumerate}

\end{document}
"""

def main():

    # Write temporary .tex file
    with open(TEX_FILE, "w") as f:
        f.write(tex_content)

    # Compile to PDF (run twice to resolve cross-references)
    for pass_num in (1, 2):
        print(f"Compiling to PDF (pass {pass_num})...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", TEX_FILE],
            cwd=REPORT_DIR,
            capture_output=True,
            text=True,
        )

    if result.returncode == 0:
        print(f"PDF generated: {PDF_FILE}")
    else:
        print("pdflatex encountered issues:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    # Clean up all LaTeX artifacts (keep only the PDF)
    for ext in [".tex", ".aux", ".log", ".out"]:
        artifact = os.path.join(REPORT_DIR, f"Cooper_Morgan_Lab6{ext}")
        if os.path.exists(artifact):
            os.remove(artifact)


if __name__ == "__main__":
    main()
