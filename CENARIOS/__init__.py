"""Pacote CENARIOS: suíte de validação funcional, visual e de estresse da garra."""

from CENARIOS import (
    test_basic_movement,
    test_dirty_surface,
    test_fog_environment,
    test_object_tracking,
    test_stress,
)

__all__ = [
    "test_basic_movement",
    "test_fog_environment",
    "test_dirty_surface",
    "test_object_tracking",
    "test_stress",
]
