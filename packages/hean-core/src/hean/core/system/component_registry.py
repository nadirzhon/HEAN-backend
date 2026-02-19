"""Generic Component Registry v2.

Provides a fully generic, topology-aware lifecycle manager for all trading
system components.  Components are registered via ``ComponentDef`` descriptors
and started/stopped in dependency order (Kahn's topological sort).  Any number
of components can be registered without modifying this module.

Key design points
-----------------
* ``Lifecycle`` protocol — every component just needs ``start()`` / ``stop()``.
* ``HealthAware`` protocol — optional; components that expose health get richer
  status reporting.
* ``ComponentDef`` — declarative descriptor that captures the factory callable,
  dependency names, config-flag name, and optional/required semantics.
* Topological sort via Kahn's algorithm; circular dependencies raise
  ``CircularDependencyError`` at registration time.
* Hot management — individual components can be restarted, enabled, or disabled
  at runtime without touching the rest of the system.
* Full backward compatibility — the three legacy attributes
  (``rl_risk_manager``, ``oracle_weighting``, ``strategy_allocator``) and all
  convenience delegation methods from v1 are preserved.

Usage example
-------------
::

    registry = ComponentRegistry(bus)

    registry.register(ComponentDef(
        name="my_widget",
        factory=lambda bus, **kw: MyWidget(bus=bus),
        deps=["oracle_weighting"],
        config_flag="my_widget_enabled",
        optional=True,
    ))

    await registry.initialize_all(initial_capital=300.0)
    await registry.start_all()
    ...
    await registry.stop_all()
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from hean.config import settings
from hean.core.bus import EventBus
from hean.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class Lifecycle:
    """Structural protocol for components that participate in system lifecycle.

    Any object that implements ``start()`` and ``stop()`` as async coroutines
    satisfies this protocol — no explicit inheritance required.
    """

    async def start(self) -> None:  # pragma: no cover
        """Bring the component to an operational state."""
        ...

    async def stop(self) -> None:  # pragma: no cover
        """Cleanly release resources and stop the component."""
        ...


class HealthAware:
    """Structural protocol for components that can report their own health.

    Returns one of the canonical strings: ``"healthy"``, ``"degraded"``,
    or ``"stopped"``.
    """

    def health_status(self) -> str:  # pragma: no cover
        """Return the current health string."""
        return "healthy"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CircularDependencyError(ValueError):
    """Raised when a cycle is detected in the component dependency graph."""


# ---------------------------------------------------------------------------
# Component definition
# ---------------------------------------------------------------------------


@dataclass
class ComponentDef:
    """Declarative descriptor for a managed component.

    Attributes
    ----------
    name:
        Unique identifier for the component within the registry.  Used as the
        dependency reference key and for logging.
    factory:
        Callable that creates the component instance.  Receives keyword
        arguments forwarded from ``initialize_all(**kwargs)`` plus ``bus``.
        Signature: ``factory(bus: EventBus, **kwargs) -> Lifecycle``.
    deps:
        Names of other components that must be started before this one.
    enabled:
        When ``False`` the component is never initialised, regardless of any
        config flag.  Defaults to ``True``; the config flag is the preferred
        runtime toggle.
    config_flag:
        Name of the ``HEANSettings`` attribute to check.  If the attribute
        evaluates to a falsy value the component is skipped.  ``None`` means
        "always enabled" (unless ``enabled=False``).
    optional:
        When ``True`` (the default), an exception during initialisation or
        start is logged but does not propagate.  When ``False``, the exception
        is re-raised and will abort system startup.
    """

    name: str
    factory: Callable[..., Any]
    deps: list[str] = field(default_factory=list)
    enabled: bool = True
    config_flag: str | None = None
    optional: bool = True


# ---------------------------------------------------------------------------
# Internal component state
# ---------------------------------------------------------------------------


@dataclass
class _ComponentState:
    """Runtime state tracked per registered component."""

    defn: ComponentDef
    instance: Any = None          # The live object; None if not yet initialised
    initialised: bool = False
    running: bool = False
    failed: bool = False           # True if the last init/start attempt failed
    error: str | None = None       # Last error message


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ComponentRegistry:
    """Generic, topology-aware component lifecycle manager.

    Components are registered via ``ComponentDef`` objects (or the convenience
    ``register_simple`` shorthand) and managed through a standard
    ``initialize_all → start_all → stop_all`` lifecycle.

    The registry resolves startup and shutdown order automatically using the
    ``deps`` graph declared in each ``ComponentDef``.  Components with no
    dependencies start first; components that depend on others start only after
    all their dependencies are running.

    Shutdown proceeds in the reverse of the startup order to honour dependency
    contracts.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._components: dict[str, _ComponentState] = {}
        # Startup order determined by _resolve_order(); populated during
        # initialize_all() so that stop_all() can reverse it correctly.
        self._startup_order: list[str] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, component_def: ComponentDef) -> None:
        """Register a component definition.

        Raises
        ------
        ValueError
            If a component with the same name is already registered.
        """
        if component_def.name in self._components:
            raise ValueError(
                f"Component '{component_def.name}' is already registered.  "
                "Use disable_component() / enable_component() to toggle it."
            )
        self._components[component_def.name] = _ComponentState(defn=component_def)
        logger.debug("Registered component definition: %s", component_def.name)

    def register_simple(
        self,
        name: str,
        factory: Callable[..., Any],
        deps: list[str] | None = None,
        *,
        config_flag: str | None = None,
        optional: bool = True,
        enabled: bool = True,
    ) -> None:
        """Convenience wrapper around :meth:`register`.

        Parameters mirror ``ComponentDef``; see its docstring for details.
        """
        self.register(
            ComponentDef(
                name=name,
                factory=factory,
                deps=deps or [],
                config_flag=config_flag,
                optional=optional,
                enabled=enabled,
            )
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize_all(self, **kwargs: Any) -> dict[str, bool]:
        """Initialise all registered components in dependency order.

        Parameters
        ----------
        **kwargs:
            Forwarded verbatim to every component factory alongside ``bus``.
            The v1 ``initial_capital`` keyword is passed through this way.

        Returns
        -------
        dict[str, bool]
            Mapping of component name to initialisation success.  Disabled or
            skipped components map to ``False``.
        """
        try:
            self._startup_order = self._resolve_order()
        except CircularDependencyError:
            logger.error("Circular dependency detected — aborting component initialisation")
            raise

        results: dict[str, bool] = {}

        for name in self._startup_order:
            state = self._components[name]
            defn = state.defn

            # Respect the static enabled flag first.
            if not defn.enabled:
                logger.debug("Component '%s' is disabled (enabled=False); skipping", name)
                results[name] = False
                continue

            # Check settings config flag if provided.
            if defn.config_flag is not None:
                flag_value = getattr(settings, defn.config_flag, None)
                if not flag_value:
                    logger.debug(
                        "Component '%s' skipped — config flag '%s' is falsy (%r)",
                        name,
                        defn.config_flag,
                        flag_value,
                    )
                    results[name] = False
                    continue

            # Attempt construction.
            try:
                instance = defn.factory(bus=self._bus, **kwargs)
                state.instance = instance
                state.initialised = True
                state.failed = False
                state.error = None
                results[name] = True
                logger.info("Initialised component: %s", name)
            except Exception as exc:
                state.failed = True
                state.error = str(exc)
                results[name] = False
                msg = f"Failed to initialise component '{name}': {exc}"
                if defn.optional:
                    logger.error(msg)
                else:
                    logger.critical(msg)
                    raise

        return results

    async def start_all(self) -> dict[str, bool]:
        """Start all initialised components in dependency order.

        Returns
        -------
        dict[str, bool]
            Mapping of component name to start success.
        """
        results: dict[str, bool] = {}

        for name in self._startup_order:
            state = self._components[name]
            if not state.initialised or state.instance is None:
                results[name] = False
                continue

            try:
                await state.instance.start()
                state.running = True
                state.failed = False
                state.error = None
                results[name] = True
                logger.info("Started component: %s", name)
            except Exception as exc:
                state.running = False
                state.failed = True
                state.error = str(exc)
                results[name] = False
                msg = f"Failed to start component '{name}': {exc}"
                if state.defn.optional:
                    logger.error(msg)
                else:
                    logger.critical(msg)
                    raise

        return results

    async def stop_all(self) -> None:
        """Stop all running components in reverse startup order."""
        stop_order = list(reversed(self._startup_order))

        for name in stop_order:
            state = self._components.get(name)
            if state is None or not state.running or state.instance is None:
                continue

            try:
                await state.instance.stop()
                state.running = False
                logger.info("Stopped component: %s", name)
            except Exception as exc:
                logger.error("Error stopping component '%s': %s", name, exc)
                # Mark as not running regardless; we're shutting down.
                state.running = False

    # ------------------------------------------------------------------
    # Hot management
    # ------------------------------------------------------------------

    async def restart_component(self, name: str) -> bool:
        """Stop, reinitialise, and restart a single component.

        The factory is called again with no extra kwargs (runtime restart —
        uses defaults).  If the component has dependencies they must already
        be running.

        Returns
        -------
        bool
            ``True`` if the component is running after the restart attempt.
        """
        state = self._components.get(name)
        if state is None:
            logger.warning("restart_component: unknown component '%s'", name)
            return False

        # Stop if currently running.
        if state.running and state.instance is not None:
            try:
                await state.instance.stop()
                state.running = False
                logger.info("Stopped '%s' for restart", name)
            except Exception as exc:
                logger.error("Error stopping '%s' during restart: %s", name, exc)

        # Re-initialise.
        try:
            instance = state.defn.factory(bus=self._bus)
            state.instance = instance
            state.initialised = True
            state.failed = False
            state.error = None
        except Exception as exc:
            state.failed = True
            state.error = str(exc)
            logger.error("Failed to reinitialise '%s' during restart: %s", name, exc)
            return False

        # Restart.
        try:
            await state.instance.start()
            state.running = True
            logger.info("Restarted component: %s", name)
            return True
        except Exception as exc:
            state.running = False
            state.failed = True
            state.error = str(exc)
            logger.error("Failed to start '%s' during restart: %s", name, exc)
            return False

    async def disable_component(self, name: str) -> None:
        """Stop and disable a component so it will not be auto-started.

        The component definition is retained; call :meth:`enable_component`
        to re-enable it.
        """
        state = self._components.get(name)
        if state is None:
            logger.warning("disable_component: unknown component '%s'", name)
            return

        if state.running and state.instance is not None:
            try:
                await state.instance.stop()
                state.running = False
            except Exception as exc:
                logger.error("Error stopping '%s' during disable: %s", name, exc)

        state.defn.enabled = False
        state.instance = None
        state.initialised = False
        logger.info("Disabled component: %s", name)

    async def enable_component(self, name: str) -> bool:
        """Enable and start a previously disabled component.

        Returns
        -------
        bool
            ``True`` if the component is running after the call.
        """
        state = self._components.get(name)
        if state is None:
            logger.warning("enable_component: unknown component '%s'", name)
            return False

        state.defn.enabled = True
        return await self.restart_component(name)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Return the live component instance, or ``None`` if unavailable."""
        state = self._components.get(name)
        if state is None:
            return None
        return state.instance

    def get_typed(self, name: str, cls: type[T]) -> T | None:
        """Return the component instance cast to *cls*, or ``None``.

        Performs an ``isinstance`` check; returns ``None`` if the component
        does not exist, is not initialised, or is not an instance of *cls*.
        """
        instance = self.get(name)
        if isinstance(instance, cls):
            return instance
        return None

    def is_running(self, name: str) -> bool:
        """Return ``True`` if the named component is currently running."""
        state = self._components.get(name)
        return state is not None and state.running

    def get_status(self) -> dict[str, Any]:
        """Return a full status snapshot of all registered components.

        The returned dict has the shape::

            {
                "total": int,
                "running": int,
                "failed": int,
                "components": {
                    "<name>": {
                        "enabled": bool,
                        "initialised": bool,
                        "running": bool,
                        "failed": bool,
                        "health": str | None,
                        "error": str | None,
                        "deps": list[str],
                        "optional": bool,
                    },
                    ...
                },
            }
        """
        component_status: dict[str, Any] = {}

        for name, state in self._components.items():
            health: str | None = None
            if state.instance is not None and hasattr(state.instance, "health_status"):
                try:
                    health = state.instance.health_status()
                except Exception:
                    health = "unknown"

            component_status[name] = {
                "enabled": state.defn.enabled,
                "initialised": state.initialised,
                "running": state.running,
                "failed": state.failed,
                "health": health,
                "error": state.error,
                "deps": list(state.defn.deps),
                "optional": state.defn.optional,
            }

        running_count = sum(1 for s in self._components.values() if s.running)
        failed_count = sum(1 for s in self._components.values() if s.failed)

        return {
            "total": len(self._components),
            "running": running_count,
            "failed": failed_count,
            "components": component_status,
        }

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """Return the adjacency list of the dependency graph.

        Keys are component names; values are the list of dependency names
        declared for that component.
        """
        return {name: list(state.defn.deps) for name, state in self._components.items()}

    # ------------------------------------------------------------------
    # Pre-registered defaults
    # ------------------------------------------------------------------

    def register_defaults(self) -> None:
        """Register the three canonical components from v1.

        This method is idempotent with respect to already-registered names
        (it skips rather than raising if a name is already present).  Call it
        after constructing the registry if you want the standard set without
        manual ``register()`` calls.

        Components registered
        ---------------------
        ``rl_risk_manager``
            Gated by ``settings.rl_risk_enabled``.
        ``oracle_weighting``
            Gated by ``settings.oracle_dynamic_weighting``.
        ``strategy_allocator``
            Always enabled (no config flag).
        """
        defaults: list[ComponentDef] = [
            ComponentDef(
                name="rl_risk_manager",
                factory=_make_rl_risk_manager,
                deps=[],
                config_flag="rl_risk_enabled",
                optional=True,
            ),
            ComponentDef(
                name="oracle_weighting",
                factory=_make_oracle_weighting,
                deps=[],
                config_flag="oracle_dynamic_weighting",
                optional=True,
            ),
            ComponentDef(
                name="strategy_allocator",
                factory=_make_strategy_allocator,
                deps=[],
                config_flag=None,
                optional=True,
            ),
        ]

        for defn in defaults:
            if defn.name not in self._components:
                self.register(defn)
            else:
                logger.debug(
                    "register_defaults: '%s' already registered, skipping", defn.name
                )

    # ------------------------------------------------------------------
    # Backward-compatible convenience properties
    # ------------------------------------------------------------------

    @property
    def rl_risk_manager(self) -> Any:
        """Direct access to the rl_risk_manager instance (v1 compat)."""
        return self.get("rl_risk_manager")

    @property
    def oracle_weighting(self) -> Any:
        """Direct access to the oracle_weighting instance (v1 compat)."""
        return self.get("oracle_weighting")

    @property
    def strategy_allocator(self) -> Any:
        """Direct access to the strategy_allocator instance (v1 compat)."""
        return self.get("strategy_allocator")

    # ------------------------------------------------------------------
    # Backward-compatible delegation methods (v1 public API)
    # ------------------------------------------------------------------

    def get_rl_risk_parameters(self) -> dict[str, float]:
        """Return current RL risk parameters.

        Delegates to ``rl_risk_manager.get_risk_parameters()`` when the
        component is running; returns safe defaults otherwise.
        """
        mgr = self.get("rl_risk_manager")
        if mgr is not None:
            try:
                return mgr.get_risk_parameters()
            except Exception as exc:
                logger.warning("get_rl_risk_parameters: error from component: %s", exc)

        return {
            "leverage": float(settings.max_leverage),
            "size_multiplier": 1.0,
            "stop_loss_pct": 2.0,
        }

    def get_oracle_weights(self) -> dict[str, float]:
        """Return current oracle model weights.

        Delegates to ``oracle_weighting.get_weights()`` when the component is
        running; returns the fixed canonical weights otherwise.
        """
        weighting = self.get("oracle_weighting")
        if weighting is not None:
            try:
                return weighting.get_weights()
            except Exception as exc:
                logger.warning("get_oracle_weights: error from component: %s", exc)

        return {
            "tcn": 0.40,
            "finbert": 0.20,
            "ollama": 0.20,
            "brain": 0.20,
        }

    def fuse_oracle_signals(
        self,
        tcn_signal: float | None = None,
        finbert_signal: float | None = None,
        ollama_signal: float | None = None,
        brain_signal: float | None = None,
        min_confidence: float = 0.6,
    ) -> dict[str, Any] | None:
        """Fuse oracle signals with dynamic weights.

        Delegates to ``oracle_weighting.fuse_signals()`` when the component is
        running; falls back to a simple unweighted average otherwise.

        Parameters
        ----------
        tcn_signal:
            TCN price-reversal prediction in [-1, 1].
        finbert_signal:
            FinBERT sentiment score in [-1, 1].
        ollama_signal:
            Ollama local-LLM sentiment score in [-1, 1].
        brain_signal:
            Claude Brain sentiment keyword score in [-1, 1].
        min_confidence:
            Minimum absolute combined score required to emit a signal.

        Returns
        -------
        dict | None
            Fused signal descriptor or ``None`` if confidence is below threshold.
        """
        weighting = self.get("oracle_weighting")
        if weighting is not None:
            try:
                return weighting.fuse_signals(
                    tcn_signal=tcn_signal,
                    finbert_signal=finbert_signal,
                    ollama_signal=ollama_signal,
                    brain_signal=brain_signal,
                    min_confidence=min_confidence,
                )
            except Exception as exc:
                logger.warning("fuse_oracle_signals: error from component: %s", exc)

        # Fallback: simple unweighted average of non-None sources.
        available = [
            s
            for s in (tcn_signal, finbert_signal, ollama_signal, brain_signal)
            if s is not None
        ]
        if not available:
            return None

        avg = sum(available) / len(available)
        confidence = abs(avg)

        if confidence < min_confidence:
            return None

        return {
            "direction": "buy" if avg > 0 else "sell",
            "confidence": confidence,
            "weighted_score": avg,
            "sources_used": ["fallback_average"],
            "weights": {},
        }

    def get_strategy_allocation(self, strategy_id: str) -> float:
        """Return the capital allocation for *strategy_id*.

        Delegates to ``strategy_allocator.get_allocation()`` when the
        component is running; returns 0.0 otherwise.
        """
        allocator = self.get("strategy_allocator")
        if allocator is not None:
            try:
                return float(allocator.get_allocation(strategy_id))
            except Exception as exc:
                logger.warning(
                    "get_strategy_allocation('%s'): error from component: %s",
                    strategy_id,
                    exc,
                )
        return 0.0

    def get_strategy_performance(self, strategy_id: str) -> Any:
        """Return performance metrics for *strategy_id*.

        Delegates to ``strategy_allocator.get_performance()`` when the
        component is running; returns ``None`` otherwise.
        """
        allocator = self.get("strategy_allocator")
        if allocator is not None:
            try:
                return allocator.get_performance(strategy_id)
            except Exception as exc:
                logger.warning(
                    "get_strategy_performance('%s'): error from component: %s",
                    strategy_id,
                    exc,
                )
        return None

    def register_strategies(self, strategy_ids: list[str]) -> None:
        """Register strategy IDs with the strategy allocator.

        Safe to call before the allocator is initialised; silently no-ops when
        the allocator is unavailable.
        """
        allocator = self.get("strategy_allocator")
        if allocator is not None:
            for sid in strategy_ids:
                try:
                    allocator.register_strategy(sid)
                except Exception as exc:
                    logger.warning(
                        "register_strategies: failed to register '%s': %s", sid, exc
                    )
            logger.info("Registered %d strategies with allocator", len(strategy_ids))
        else:
            logger.debug(
                "register_strategies: strategy_allocator not available, "
                "skipping %d IDs",
                len(strategy_ids),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_order(self) -> list[str]:
        """Return component names in a valid initialisation order.

        Uses Kahn's algorithm on the dependency graph.  Unknown dependency
        names are logged as warnings (the referenced component may be external
        or optional) but do not block resolution.

        Raises
        ------
        CircularDependencyError
            If a cycle exists in the declared dependency graph.
        """
        # Build in-degree map and adjacency list (dependant → dependency edges
        # already expressed as deps).  We want a start-order where every dep
        # appears *before* its dependant.
        all_names = set(self._components.keys())
        in_degree: dict[str, int] = dict.fromkeys(all_names, 0)
        # dependant[dep] = list of components that depend on dep
        dependants: dict[str, list[str]] = {name: [] for name in all_names}

        for name, state in self._components.items():
            for dep in state.defn.deps:
                if dep not in all_names:
                    logger.warning(
                        "_resolve_order: component '%s' declares dependency on "
                        "unknown component '%s' — ignoring",
                        name,
                        dep,
                    )
                    continue
                in_degree[name] += 1
                dependants[dep].append(name)

        # Kahn's algorithm.
        queue: deque[str] = deque(
            name for name, degree in in_degree.items() if degree == 0
        )
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for dependant in dependants.get(node, []):
                in_degree[dependant] -= 1
                if in_degree[dependant] == 0:
                    queue.append(dependant)

        if len(order) != len(all_names):
            cycle_nodes = [n for n in all_names if n not in order]
            raise CircularDependencyError(
                f"Circular dependency detected among components: {cycle_nodes}"
            )

        return order


# ---------------------------------------------------------------------------
# Default component factory functions
# ---------------------------------------------------------------------------
# Each factory receives (bus, **kwargs) and returns a Lifecycle instance.
# They are module-level rather than lambdas to enable clear tracebacks.


def _make_rl_risk_manager(bus: EventBus, **kwargs: Any) -> Any:
    """Factory for RLRiskManager.  Raises ImportError if deps are missing."""
    from hean.risk.rl_risk_manager import RLRiskManager  # type: ignore[import]

    return RLRiskManager(
        bus=bus,
        model_path=settings.rl_risk_model_path,
        adjustment_interval=settings.rl_risk_adjust_interval,
        enabled=settings.rl_risk_enabled,
    )


def _make_oracle_weighting(bus: EventBus, **kwargs: Any) -> Any:
    """Factory for DynamicOracleWeighting.  Raises ImportError if deps are missing."""
    from hean.core.intelligence.dynamic_oracle import DynamicOracleWeighting  # type: ignore[import]

    return DynamicOracleWeighting(bus=bus)


def _make_strategy_allocator(bus: EventBus, **kwargs: Any) -> Any:
    """Factory for StrategyAllocator.  Raises ImportError if deps are missing."""
    from hean.strategies.manager import StrategyAllocator  # type: ignore[import]

    initial_capital: float = float(kwargs.get("initial_capital", 10_000.0))
    return StrategyAllocator(
        bus=bus,
        initial_capital=initial_capital,
        rebalance_interval=300,
        min_allocation_pct=0.05,
        max_allocation_pct=0.40,
    )


# ---------------------------------------------------------------------------
# Module-level singleton (backward compatible with v1)
# ---------------------------------------------------------------------------

_registry: ComponentRegistry | None = None


def get_component_registry() -> ComponentRegistry | None:
    """Return the process-wide ComponentRegistry singleton, or ``None``."""
    return _registry


def set_component_registry(registry: ComponentRegistry) -> None:
    """Install *registry* as the process-wide ComponentRegistry singleton."""
    global _registry
    _registry = registry
