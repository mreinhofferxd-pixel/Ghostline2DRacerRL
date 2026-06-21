"""Agent-driven physics evaluation.

The game core is a deterministic, headless, pygame-free simulation, so a coding
LLM can *measure* the physics instead of eyeballing a rendered window. This
package turns qualitative drift criteria into reproducible scripted
experiments that return numbers (see ``harness``). Run the report headless with:

    python -m momentum_lab.eval
"""
