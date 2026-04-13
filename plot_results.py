"""
Plot FL + DP Training Results
Uses actual data from training_history.json
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Load real results ──────────────────────────────────────────
with open('results/training_history.json') as f:
    history = json.load(f)

rounds = [h['round'] for h in history]
accuracy = [round(h['test_accuracy'] * 100, 2) for h in history]
loss = [round(h['test_loss'], 4) for h in history]

# ── Figure with 2 subplots ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#f8f9fa')

# ── Colors ─────────────────────────────────────────────────────
dp_color = '#667eea'
loss_color = '#f87171'

# ══════════════════════════════════════════════
# Plot 1: Accuracy per Round
# ══════════════════════════════════════════════
ax1.set_facecolor('white')
ax1.plot(rounds, accuracy, color=dp_color, linewidth=2.5,
         marker='o', markersize=7, markerfacecolor='white',
         markeredgecolor=dp_color, markeredgewidth=2, label='FL + DP')

# Shade under the curve
ax1.fill_between(rounds, accuracy, alpha=0.1, color=dp_color)

# Annotate best point
best_round = rounds[accuracy.index(max(accuracy))]
best_acc = max(accuracy)
ax1.annotate(f'Best: {best_acc}%',
             xy=(best_round, best_acc),
             xytext=(best_round - 2, best_acc - 4),
             fontsize=9, color=dp_color, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=dp_color, lw=1.5))

# Annotate round 1 jump
ax1.annotate(f'Round 1: {accuracy[1]}%',
             xy=(rounds[1], accuracy[1]),
             xytext=(rounds[1] + 0.5, accuracy[1] - 5),
             fontsize=8, color='#374151',
             arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.2))

ax1.set_title('Model Accuracy per Round\n(Federated Learning + Differential Privacy)',
              fontsize=13, fontweight='bold', color='#1a1a2e', pad=15)
ax1.set_xlabel('Training Round', fontsize=11, color='#374151')
ax1.set_ylabel('Test Accuracy (%)', fontsize=11, color='#374151')
ax1.set_xlim(-0.3, max(rounds) + 0.3)
ax1.set_ylim(0, 105)
ax1.set_xticks(rounds)
ax1.tick_params(colors='#374151')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add epsilon annotation
ax1.text(0.98, 0.05,
         'Privacy: ε = 0.06 / 5.0\nNoise Multiplier: 1.0',
         transform=ax1.transAxes,
         fontsize=8, color='#667eea',
         ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#eef2ff', edgecolor='#667eea', alpha=0.8))

ax1.legend(fontsize=10, loc='lower right')

# ══════════════════════════════════════════════
# Plot 2: Loss per Round
# ══════════════════════════════════════════════
ax2.set_facecolor('white')
ax2.plot(rounds, loss, color=loss_color, linewidth=2.5,
         marker='s', markersize=7, markerfacecolor='white',
         markeredgecolor=loss_color, markeredgewidth=2, label='FL + DP')

ax2.fill_between(rounds, loss, alpha=0.1, color=loss_color)

# Annotate best (lowest) loss
min_loss = min(loss)
min_round = rounds[loss.index(min_loss)]
ax2.annotate(f'Best: {min_loss}',
             xy=(min_round, min_loss),
             xytext=(min_round - 2, min_loss + 0.05),
             fontsize=9, color=loss_color, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=loss_color, lw=1.5))

ax2.set_title('Model Loss per Round\n(Federated Learning + Differential Privacy)',
              fontsize=13, fontweight='bold', color='#1a1a2e', pad=15)
ax2.set_xlabel('Training Round', fontsize=11, color='#374151')
ax2.set_ylabel('Test Loss', fontsize=11, color='#374151')
ax2.set_xlim(-0.3, max(rounds) + 0.3)
ax2.set_xticks(rounds)
ax2.tick_params(colors='#374151')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(fontsize=10, loc='upper right')

# ── Summary box ────────────────────────────────────────────────
summary = (
    f"Final Accuracy: {accuracy[-1]}%  |  "
    f"Best Accuracy: {max(accuracy)}%  |  "
    f"Final Loss: {loss[-1]}  |  "
    f"Rounds: {max(rounds)}  |  "
    f"Privacy Budget Used: ε=0.06/5.0"
)
fig.text(0.5, 0.01, summary,
         ha='center', fontsize=9, color='#4b5563',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef2ff',
                   edgecolor='#667eea', alpha=0.8))

plt.suptitle('Fertility Risk Prediction — FL + DP Training Results',
             fontsize=15, fontweight='bold', color='#1a1a2e', y=1.02)

plt.tight_layout(rect=[0, 0.06, 1, 1])

# ── Save ───────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
plt.savefig('results/training_results_graph.png',
            dpi=150, bbox_inches='tight',
            facecolor='#f8f9fa')
print("✓ Graph saved to results/training_results_graph.png")
plt.show()
