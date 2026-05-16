"""M7-Evolve: ASI-Evolve-style autonomous improvement loop.

Inspired by GAIR-NLP/ASI-Evolve. Four key ideas adapted for M7:

  1. **Trial database** — every iteration is a structured record:
     motivation, action (mutation), metrics, lesson, parent_id. Stored
     persistently as JSON so successful and failed trials both inform
     future rounds.

  2. **Cognition store** — accumulates "lessons" from prior trials
     ("biasing toward flux rows hurts count MSE", "lr=1e-3 too high
     causes divergence"). The Researcher consults these to avoid
     re-trying known bad moves.

  3. **MAP-Elites parent selection** — maintains a population of diverse
     high-performing M7 variants across behaviour dimensions
     (bias_strength × target_category × learning_rate). New trials
     branch from a randomly-chosen cell in the grid, encouraging
     exploration in unfilled regions.

  4. **Researcher / Engineer / Analyzer separation** — three small
     components with clean responsibilities. The Researcher is
     rule-based by default but has an optional LLM hook.

Compared to iterate.py (linear chain), this maintains a TREE of
variants and chooses where to branch each round. Converges slower per
round but reaches better final variants by exploring more of the
modification space.

Usage:
    python -m cell_sim.lgnn.self_improve.evolve \\
        --initial /content/drive/.../m7_7_best.pt \\
        --output  /content/drive/.../evolve \\
        --n_rounds 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cell_sim.lgnn.self_improve.iterate import (
    analyze_model_weaknesses,
    compute_full_val_metrics,
    fine_tune_on_weaknesses,
    load_m7_with_state,
)
from cell_sim.lgnn.data.species_graph import load_species_graph


# ============================================================
# Trial — one structured experimental record
# ============================================================
@dataclass
class Trial:
    trial_id: int
    parent_id: Optional[int]
    round: int
    motivation: str
    action: Dict[str, Any]
    metrics: Dict[str, float]
    delta_composite: float
    accepted: bool
    lesson: str
    timestamp: float
    checkpoint_path: str
    behavior_cell: Tuple[int, int, int] = (0, 0, 0)


# ============================================================
# Cognition store — accumulated heuristics + lessons
# ============================================================
class CognitionStore:
    """Prior knowledge + lessons distilled from past trials."""

    SEED_HEURISTICS = [
        # Hardcoded biology priors for the Researcher to consider
        'aminoacyl-tRNA-synthetases (genes 0001-0030) are essential, '
        'high impact under KO; F_* flux rows for tRNA charging matter.',
        'transport reactions (M_*_e -> M_*_c) are sensitive to medium '
        'features; if those are decorative, target them via biased '
        'sampling on transport-related rows.',
        'PINN-mapped rows (~140 of 8572) are constrained by Δx=S·v; '
        'most prediction error sits in the residual (non-PINN) rows.',
        'val_count plateaus at ~0.0716 from stochastic noise; do not '
        'chase that metric, target val_rollout_200_mse and per-species '
        'worst-case instead.',
        'fine-tunes with lr > 5e-4 tend to destabilize the encoder; '
        'safe range is 5e-5 to 3e-4.',
        'biased sampling toward weak timesteps hurts uniform performance '
        'if bias_strength > 0.85; sweet spot 0.6-0.75.',
    ]

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self.lessons: List[str] = list(self.SEED_HEURISTICS)
        if persist_path and os.path.exists(persist_path):
            try:
                with open(persist_path) as f:
                    data = json.load(f)
                self.lessons = data.get('lessons', list(self.SEED_HEURISTICS))
            except Exception:
                pass

    def add_lesson(self, lesson: str):
        if lesson and lesson not in self.lessons:
            self.lessons.append(lesson)
        self.save()

    def save(self):
        if self.persist_path:
            with open(self.persist_path, 'w') as f:
                json.dump({'lessons': self.lessons}, f, indent=2)

    def relevant_lessons(self, k: int = 5) -> List[str]:
        """Return the K most recent lessons (most relevant to current run)."""
        return self.lessons[-k:]


# ============================================================
# Trial database with parent-selection algorithms
# ============================================================
class TrialDatabase:
    def __init__(self, persist_path: Optional[str] = None):
        self.trials: List[Trial] = []
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            try:
                with open(persist_path) as f:
                    raw = json.load(f)
                self.trials = [Trial(**{**t,
                                        'behavior_cell': tuple(t.get('behavior_cell', (0, 0, 0)))})
                               for t in raw]
            except Exception:
                pass

    def add(self, trial: Trial):
        trial.trial_id = len(self.trials)
        self.trials.append(trial)
        self.save()

    def save(self):
        if self.persist_path:
            with open(self.persist_path, 'w') as f:
                json.dump([asdict(t) for t in self.trials], f, indent=2,
                          default=str)

    def accepted(self) -> List[Trial]:
        return [t for t in self.trials if t.accepted]

    def best(self) -> Optional[Trial]:
        acc = self.accepted()
        if not acc:
            return None
        return min(acc, key=lambda t: t.metrics.get('composite', float('inf')))

    # ---- Parent selection strategies ----
    def select_parent_ucb1(self, exploration_c: float = 1.4) -> Optional[Trial]:
        """UCB1: pick parent that maximizes mean improvement + sqrt(2 ln N / n).
        Promotes both high-performers and less-explored branches."""
        acc = self.accepted()
        if not acc:
            return None
        # Count how many times each trial has been a parent
        child_counts = {}
        for t in self.trials:
            if t.parent_id is not None:
                child_counts[t.parent_id] = child_counts.get(t.parent_id, 0) + 1
        N = sum(child_counts.values()) + 1
        best = None
        best_score = -float('inf')
        for t in acc:
            n = child_counts.get(t.trial_id, 0) + 1
            mean_imp = -t.metrics.get('composite', 1.0)
            ucb = mean_imp + exploration_c * math.sqrt(2 * math.log(N) / n)
            if ucb > best_score:
                best_score = ucb
                best = t
        return best

    def select_parent_map_elites(self, grid_shape=(3, 4, 3)) -> Optional[Trial]:
        """MAP-Elites: pick best from each cell of a behaviour grid,
        then sample one cell uniformly. Encourages diversity."""
        acc = self.accepted()
        if not acc:
            return None
        # Group by behavior_cell, keep best in each
        cell_best: Dict[Tuple[int, int, int], Trial] = {}
        for t in acc:
            c = t.behavior_cell
            if c not in cell_best or t.metrics.get('composite', 1.0) < cell_best[c].metrics.get('composite', 1.0):
                cell_best[c] = t
        if not cell_best:
            return None
        return random.choice(list(cell_best.values()))

    def select_parent(self, strategy: str = 'map_elites') -> Optional[Trial]:
        if strategy == 'ucb1':
            return self.select_parent_ucb1()
        if strategy == 'map_elites':
            return self.select_parent_map_elites()
        if strategy == 'greedy':
            return self.best()
        return random.choice(self.accepted()) if self.accepted() else None


# ============================================================
# Researcher — proposes the next mutation
# ============================================================
class Researcher:
    """Rule-based action proposer. Picks a mutation type uniformly,
    then samples parameters with mild bias away from values seen to
    fail in past trials.

    To swap in an LLM-driven Researcher, subclass and override
    `propose_action`. The LLM receives the cognition lessons and
    last-N trial outcomes as context."""

    ACTIONS = [
        'tune_lr',
        'tune_bias_strength',
        'target_category',
        'tune_steps',
        'tune_weight_count',
        'tune_weight_flux',
        'add_weight_decay',
        'tune_aux_weight',          # only meaningful for M7.8+
    ]

    def __init__(self, cognition: CognitionStore, seed: int = 42):
        self.cognition = cognition
        self.rng = random.Random(seed)

    def _failed_values(self, db: TrialDatabase, key: str) -> List[Any]:
        """Values for `key` that appeared in rejected trials."""
        return [t.action.get(key)
                for t in db.trials
                if not t.accepted and key in t.action]

    def propose_action(self, db: TrialDatabase, parent: Optional[Trial]
                       ) -> Tuple[Dict[str, Any], str, Tuple[int, int, int]]:
        """Return (action_dict, motivation_string, behavior_cell)."""
        choice = self.rng.choice(self.ACTIONS)
        failed = self._failed_values(db, choice)

        # Default base action
        action = {
            'fine_tune_steps': 2000,
            'bias_strength': 0.7,
            'lr': 1e-4,
            'weight_count': 0.05,
            'weight_flux': 1.0,
            'weight_cumulative': 0.5,
            'weight_decay': 1e-4,
            'aux_loss_weight': 0.5,
            'target_category': 'all',
        }

        motivation_parts = []
        if choice == 'tune_lr':
            cands = [3e-5, 1e-4, 3e-4]
            cands = [c for c in cands if c not in failed]
            if not cands:
                cands = [1e-4]
            action['lr'] = self.rng.choice(cands)
            motivation_parts.append(f'try lr={action["lr"]:.0e}')
        elif choice == 'tune_bias_strength':
            cands = [0.5, 0.6, 0.7, 0.8]
            cands = [c for c in cands if c not in failed]
            action['bias_strength'] = self.rng.choice(cands) if cands else 0.7
            motivation_parts.append(
                f'bias_strength={action["bias_strength"]:.1f}')
        elif choice == 'target_category':
            cands = ['count', 'flux', 'cum', 'balanced', 'all']
            cands = [c for c in cands if c not in failed[-3:]]
            action['target_category'] = self.rng.choice(cands)
            motivation_parts.append(
                f'focus on {action["target_category"]} rows')
        elif choice == 'tune_steps':
            action['fine_tune_steps'] = self.rng.choice([1000, 2000, 4000])
            motivation_parts.append(f'{action["fine_tune_steps"]} fine-tune steps')
        elif choice == 'tune_weight_count':
            action['weight_count'] = self.rng.choice([0.05, 0.1, 0.25, 0.5])
            motivation_parts.append(f'weight_count={action["weight_count"]}')
        elif choice == 'tune_weight_flux':
            action['weight_flux'] = self.rng.choice([0.5, 1.0, 2.0])
            motivation_parts.append(f'weight_flux={action["weight_flux"]}')
        elif choice == 'add_weight_decay':
            action['weight_decay'] = self.rng.choice([0.0, 1e-5, 1e-4, 1e-3])
            motivation_parts.append(f'weight_decay={action["weight_decay"]:.0e}')
        elif choice == 'tune_aux_weight':
            action['aux_loss_weight'] = self.rng.choice([0.1, 0.3, 0.5, 1.0])
            motivation_parts.append(
                f'aux_loss_weight={action["aux_loss_weight"]}')

        # Behaviour cell for MAP-Elites: discretise three knobs
        bs_idx = min(2, max(0, int((action['bias_strength'] - 0.5) / 0.15)))
        tc_idx = ['count', 'flux', 'cum', 'balanced', 'all'].index(
            action['target_category']) % 4
        lr_idx = min(2, max(0, int(round(math.log10(action['lr']) + 5))))
        behavior_cell = (bs_idx, tc_idx, lr_idx)

        parent_note = (
            f'branching from trial {parent.trial_id} '
            f'(composite {parent.metrics.get("composite", float("nan")):.4f})'
            if parent is not None else 'baseline branch'
        )

        # Inject one recent lesson into the motivation
        lessons = self.cognition.relevant_lessons(k=2)
        lesson_str = lessons[-1] if lessons else ''

        motivation = (
            f'{parent_note}. Hypothesis: '
            f'{" + ".join(motivation_parts)}. '
            f'Heuristic context: "{lesson_str[:80]}..."'
        )
        return action, motivation, behavior_cell


# ============================================================
# Engineer — applies a mutation and runs fine-tune
# ============================================================
class Engineer:
    def __init__(self, train_data, val_data, sigma, sg, row_names,
                 output_dir: str, device: str = 'cuda'):
        self.train_data = train_data
        self.val_data = val_data
        self.sigma = sigma
        self.sg = sg
        self.row_names = row_names
        self.output_dir = output_dir
        self.device = device

    def execute(
        self,
        parent_checkpoint: str,
        action: Dict[str, Any],
        round_idx: int,
        trial_id: int,
    ) -> Tuple[str, Dict[str, float]]:
        """Apply mutation, fine-tune, return (checkpoint_path, metrics)."""
        model, cfg, fidx = load_m7_with_state(
            parent_checkpoint, self.sg, self.row_names, self.device,
        )
        # Profile weaknesses from current model
        weakness = analyze_model_weaknesses(
            model, self.val_data, self.sigma, self.row_names,
            t_step_stride=100,
        )
        # Apply target_category: filter weak species/rows by prefix
        cat = action.get('target_category', 'all')
        if cat in {'count', 'flux', 'cum'}:
            prefix_map = {'count': ['G_', 'R_', 'P_', 'M_'],
                          'flux': ['F_'], 'cum': ['C_']}
            prefixes = prefix_map[cat]
            new_idx = []
            for sp_idx, sp_name in zip(weakness['worst_species_idx'],
                                        weakness['worst_species_names']):
                if any(sp_name.startswith(p) for p in prefixes):
                    new_idx.append(sp_idx)
            if new_idx:
                weakness['worst_species_idx'] = new_idx[:50]

        # Fine-tune with mutated hyperparams
        model = fine_tune_on_weaknesses(
            model, self.train_data, self.sigma, self.val_data,
            self.row_names, fidx, weakness,
            n_steps=int(action.get('fine_tune_steps', 2000)),
            batch_size=256,
            lr=float(action.get('lr', 1e-4)),
            bias_strength=float(action.get('bias_strength', 0.7)),
            weight_count=float(action.get('weight_count', 0.05)),
            weight_flux=float(action.get('weight_flux', 1.0)),
            weight_cumulative=float(action.get('weight_cumulative', 0.5)),
            log_every=500,
        )
        # Evaluate
        metrics = compute_full_val_metrics(
            model, self.val_data, self.sigma, self.row_names,
        )
        # Save
        ckpt_path = f'{self.output_dir}/m7_evolve_t{trial_id:03d}.pt'
        torch.save({
            'state_dict': model.state_dict(),
            'cfg': cfg, 'architecture': 'M7_hybrid',
            'reaction_ids': [], 'flux_indices': fidx.cpu().tolist(),
            'self_improve_metrics': metrics,
            'evolve_trial_id': trial_id, 'evolve_round': round_idx,
            'evolve_action': action,
        }, ckpt_path)
        return ckpt_path, metrics


# ============================================================
# Analyzer — converts outcomes into reusable lessons
# ============================================================
class Analyzer:
    """Extract a textual lesson from each trial outcome.

    The lesson distillation here is simple rule-based: it looks at
    which action parameters moved which metrics. A future version
    could route this through an LLM for richer prose."""

    def extract_lesson(
        self, trial: Trial, parent_metrics: Dict[str, float],
    ) -> str:
        new = trial.metrics
        delta_total = new['val_singlestep_mse'] - parent_metrics.get(
            'val_singlestep_mse', float('inf'))
        delta_rollout = new['val_rollout_200_mse'] - parent_metrics.get(
            'val_rollout_200_mse', float('inf'))
        action = trial.action
        verb = 'helped' if trial.accepted else 'hurt'
        # Identify which mutation was most distinctive vs defaults
        defaults = {'lr': 1e-4, 'bias_strength': 0.7,
                    'fine_tune_steps': 2000,
                    'weight_count': 0.05, 'weight_flux': 1.0,
                    'target_category': 'all', 'aux_loss_weight': 0.5}
        nondefault = {k: v for k, v in action.items() if defaults.get(k) != v}
        if not nondefault:
            return f'Default action {verb} (Δss={delta_total:+.4f}, Δrollout={delta_rollout:+.4f})'
        feature = list(nondefault.items())[0]
        return (f'{feature[0]}={feature[1]} {verb} (Δss={delta_total:+.4f}, '
                f'Δrollout={delta_rollout:+.4f}, composite delta={trial.delta_composite:+.4f})')


# ============================================================
# Orchestrator
# ============================================================
class M7Evolve:
    def __init__(
        self,
        initial_checkpoint: str,
        train_data, val_data, sigma, sg, row_names,
        output_dir: str,
        device: str = 'cuda',
        parent_strategy: str = 'map_elites',
        seed: int = 42,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.initial_checkpoint = initial_checkpoint
        self.output_dir = output_dir
        self.device = device
        self.parent_strategy = parent_strategy
        self.rng = random.Random(seed)

        self.cognition = CognitionStore(f'{output_dir}/cognition.json')
        self.database = TrialDatabase(f'{output_dir}/trials.json')
        self.researcher = Researcher(self.cognition, seed=seed)
        self.engineer = Engineer(train_data, val_data, sigma, sg, row_names,
                                  output_dir, device)
        self.analyzer = Analyzer()
        self.train_data = train_data
        self.val_data = val_data
        self.sigma = sigma
        self.sg = sg
        self.row_names = row_names

    def _seed_baseline(self):
        """Add the initial checkpoint as trial 0 if database is empty."""
        if self.database.trials:
            print(f'Database has {len(self.database.trials)} prior trials; '
                  f'resuming')
            return
        model, cfg, fidx = load_m7_with_state(
            self.initial_checkpoint, self.sg, self.row_names, self.device,
        )
        print('Baseline: evaluating initial checkpoint...')
        metrics = compute_full_val_metrics(
            model, self.val_data, self.sigma, self.row_names,
        )
        baseline = Trial(
            trial_id=0, parent_id=None, round=0,
            motivation='baseline', action={},
            metrics=metrics, delta_composite=0.0,
            accepted=True, lesson='initial baseline',
            timestamp=time.time(),
            checkpoint_path=self.initial_checkpoint,
        )
        self.database.add(baseline)
        del model

    def evolve(self, n_rounds: int = 20, early_stop_streak: int = 5):
        """Run N rounds of researcher / engineer / analyzer."""
        self._seed_baseline()
        print(f'\n{"=" * 60}')
        print(f'M7-Evolve: {n_rounds} rounds, parent strategy={self.parent_strategy}')
        print(f'{"=" * 60}')

        streak_rejected = 0
        for r in range(1, n_rounds + 1):
            print(f'\n--- Round {r}/{n_rounds} ---')
            # Researcher: pick parent + propose mutation
            parent = self.database.select_parent(self.parent_strategy)
            action, motivation, behavior = self.researcher.propose_action(
                self.database, parent,
            )
            parent_checkpoint = (parent.checkpoint_path if parent
                                 else self.initial_checkpoint)
            print(f'Researcher: {motivation}')
            print(f'  parent={parent.trial_id if parent else "init"}  '
                  f'action={action}  behavior={behavior}')

            # Engineer: apply mutation + evaluate
            trial_id = len(self.database.trials)
            ckpt_path, metrics = self.engineer.execute(
                parent_checkpoint, action, r, trial_id,
            )
            parent_metrics = parent.metrics if parent else metrics
            delta_comp = (metrics['composite']
                          - parent_metrics.get('composite', float('inf')))

            # Acceptance: better than parent AND better than global best so far
            best = self.database.best()
            best_comp = (best.metrics['composite'] if best
                         else float('inf'))
            accepted = (delta_comp < -1e-5 and
                        metrics['composite'] < best_comp + 1e-5)
            print(f'Engineer: composite={metrics["composite"]:.4f}  '
                  f'(Δ {delta_comp:+.4f}  best={best_comp:.4f})  '
                  f'{"ACCEPTED" if accepted else "rejected"}')

            # Analyzer: extract lesson
            trial = Trial(
                trial_id=trial_id,
                parent_id=parent.trial_id if parent else None,
                round=r,
                motivation=motivation, action=action,
                metrics=metrics,
                delta_composite=delta_comp,
                accepted=accepted,
                lesson='',
                timestamp=time.time(),
                checkpoint_path=ckpt_path,
                behavior_cell=behavior,
            )
            lesson = self.analyzer.extract_lesson(trial, parent_metrics)
            trial.lesson = lesson
            self.cognition.add_lesson(lesson)
            self.database.add(trial)
            print(f'Analyzer: {lesson}')

            # Don't keep rejected checkpoints on disk
            if not accepted:
                streak_rejected += 1
                try:
                    if os.path.exists(ckpt_path):
                        os.remove(ckpt_path)
                except OSError:
                    pass
            else:
                streak_rejected = 0

            if streak_rejected >= early_stop_streak:
                print(f'\nEarly stop: {streak_rejected} consecutive '
                      f'rejected rounds.')
                break

        best = self.database.best()
        print(f'\n{"=" * 60}')
        print('Evolve complete.')
        print(f'{"=" * 60}')
        if best:
            print(f'Best trial: {best.trial_id}  '
                  f'composite={best.metrics["composite"]:.4f}  '
                  f'checkpoint={best.checkpoint_path}')
        n_acc = len(self.database.accepted()) - 1   # excl baseline
        n_rej = len([t for t in self.database.trials if not t.accepted])
        print(f'Rounds: {n_acc} accepted, {n_rej} rejected')
        return best


# ============================================================
# Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial', required=True)
    parser.add_argument('--output',  required=True)
    parser.add_argument('--n_rounds', type=int, default=20)
    parser.add_argument('--parent_strategy',
                        choices=['ucb1', 'map_elites', 'greedy', 'random'],
                        default='map_elites')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--species_graph',
                        default='/content/drive/MyDrive/cell_count_dynamics/species_graph_v7.pt')
    parser.add_argument('--train_cache',
                        default='/content/drive/MyDrive/cell_count_dynamics/preload_train.pt')
    parser.add_argument('--val_cache',
                        default='/content/drive/MyDrive/cell_count_dynamics/preload_val.pt')
    parser.add_argument('--sigma',
                        default='/content/drive/MyDrive/cell_count_dynamics/sigma_train_reps_1to49.pt')
    args = parser.parse_args()

    print('Loading shared resources...')
    sg = load_species_graph(args.species_graph)
    row_names = list(sg.row_names)
    train_data = torch.load(args.train_cache, map_location=args.device,
                             weights_only=False)
    val_data = torch.load(args.val_cache, map_location=args.device,
                           weights_only=False)
    sigma = torch.load(args.sigma, map_location=args.device,
                        weights_only=False)
    print(f'  train: {tuple(train_data.shape)}  val: {tuple(val_data.shape)}')

    evolver = M7Evolve(
        initial_checkpoint=args.initial,
        train_data=train_data, val_data=val_data, sigma=sigma,
        sg=sg, row_names=row_names,
        output_dir=args.output,
        device=args.device,
        parent_strategy=args.parent_strategy,
        seed=args.seed,
    )
    evolver.evolve(n_rounds=args.n_rounds)


if __name__ == '__main__':
    main()
