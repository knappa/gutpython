import functools
import itertools
import logging
import math
from typing import Callable, Final, Iterable, cast

import dill
import h5py
import numpy as np
from attr import Factory, define, field, fields
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)
DEFAULT_SIZE = 200


# noinspection PyUnusedLocal
def _constant_function(t: float, *, s: int | float) -> float | int:
    return s


def scalar_to_function(
    s: int | float | Callable[[float], int | float],
) -> Callable[[float], int | float]:
    if isinstance(s, int) or isinstance(s, float):
        return functools.partial(_constant_function, s=s)
    else:
        return s


@define(kw_only=True, frozen=False)
class GutPython:
    GRID_WIDTH: int = field(default=100, metadata={"type": "parameter"})
    GRID_HEIGHT: int = field(default=1, metadata={"type": "parameter"})
    MAX_BIFIDOS: int = field(default=DEFAULT_SIZE, metadata={"type": "bookkeeping"})
    MAX_DESULFOS: int = field(default=DEFAULT_SIZE, metadata={"type": "bookkeeping"})
    MAX_CLOSTS: int = field(default=DEFAULT_SIZE, metadata={"type": "bookkeeping"})
    MAX_BACTEROIDS: int = field(default=DEFAULT_SIZE, metadata={"type": "bookkeeping"})

    rng: np.random.Generator = field(
        factory=lambda: np.random.default_rng(), metadata={"type": "bookkeeping"}
    )

    ######################################################################
    # parameters from interface

    max_stuck_chance: float = field(
        default=50, metadata={"type": "parameter"}
    )  # TODO: understand units. percent?
    low_stuck_bound: float = field(
        default=2, metadata={"type": "parameter"}
    )  # TODO: understand units. percent?
    unstuck_chance: float = field(
        default=10, metadata={"type": "parameter"}
    )  # TODO: understand units. percent?
    mid_stuck_conc: float = field(
        default=10.0, metadata={"type": "parameter"}
    )  # TODO: understand units. percent?
    seed_chance: float = field(
        default=5.0, metadata={"type": "parameter"}
    )  # TODO: understand units. percent?
    seed_percent: float = field(default=5.0, metadata={"type": "parameter"})

    absorption: float = field(
        default=0.0, metadata={"type": "parameter"}
    )  # TODO: understand units.
    reserve_fraction: float = field(
        default=0.0, metadata={"type": "parameter"}
    )  # TODO: understand units.

    init_num_bifidos: int = field(
        default=23562, metadata={"type": "init_parameter"}
    )  # TODO: uhh? Seems awfully specific.
    init_num_bacteroids: int = field(
        default=5490, metadata={"type": "init_parameter"}
    )  # TODO: uhh? Seems awfully specific.
    init_num_closts: int = field(
        default=921, metadata={"type": "init_parameter"}
    )  # TODO: uhh? Seems awfully specific.
    init_num_desulfos: int = field(default=70, metadata={"type": "init_parameter"})

    bifido_lactate_production: float = field(default=0.005, metadata={"type": "parameter"})

    flow_dist: float = field(default=0.28, metadata={"type": "parameter"})

    bifido_doub: int = field(default=330, metadata={"type": "parameter"})
    desulfo_doub: int = field(default=330, metadata={"type": "parameter"})
    bacteroid_doub: int = field(default=330, metadata={"type": "parameter"})
    clost_doub: int = field(default=330, metadata={"type": "parameter"})

    bifido_flow_const: float = field(default=1.0, metadata={"type": "parameter"})
    desulfo_flow_const: float = field(default=1.0, metadata={"type": "parameter"})
    clost_flow_const: float = field(default=1.0, metadata={"type": "parameter"})
    bacteroid_flow_const: float = field(default=1.0, metadata={"type": "parameter"})

    ######################################################################
    # in flow parameters (controls)

    tick_in_flow: int = field(default=480, metadata={"type": "parameter"})

    inulin_inflow: Callable[[int], float] = field(
        default=lambda t: 10.0, converter=scalar_to_function, metadata={"type": "control"}
    )
    fo_inflow: Callable[[int], float] = field(
        default=lambda t: 25.0, converter=scalar_to_function, metadata={"type": "control"}
    )
    lactose_inflow: Callable[[int], float] = field(
        default=lambda t: 15.0, converter=scalar_to_function, metadata={"type": "control"}
    )
    lactate_inflow: Callable[[int], float] = field(
        default=lambda t: 0.0, converter=scalar_to_function, metadata={"type": "control"}
    )
    glucose_inflow: Callable[[int], float] = field(
        default=lambda t: 30.0, converter=scalar_to_function, metadata={"type": "control"}
    )
    cs_inflow: Callable[[int], float] = field(
        default=lambda t: 0.1, converter=scalar_to_function, metadata={"type": "control"}
    )

    in_conc_bacteroids: Callable[[int], int | float] = field(
        default=lambda t: 0, converter=scalar_to_function, metadata={"type": "control"}
    )
    in_conc_bifidos: Callable[[int], int | float] = field(
        default=lambda t: 0, converter=scalar_to_function, metadata={"type": "control"}
    )
    in_conc_closts: Callable[[int], int | float] = field(
        default=lambda t: 0, converter=scalar_to_function, metadata={"type": "control"}
    )
    in_conc_desulfos: Callable[[int], int | float] = field(
        default=lambda t: 0, converter=scalar_to_function, metadata={"type": "control"}
    )

    ######################################################################
    # other parameters

    absorption_constant: Final[float] = field(default=0.723823204, metadata={"type": "parameter"})

    ######################################################################
    # other globals

    ticks: int = field(default=0, metadata={"type": "parameter"})

    ######################################################################
    # measurements

    sim_terminated: bool = field(default=False, init=False, metadata={"type": "bookkeeping"})

    excreted_bacteroids: int = field(default=0, init=False, metadata={"type": "measurement"})
    excreted_bifidos: int = field(default=0, init=False, metadata={"type": "measurement"})
    excreted_closts: int = field(default=0, init=False, metadata={"type": "measurement"})
    excreted_desulfos: int = field(default=0, init=False, metadata={"type": "measurement"})
    # TODO: these are all live cells, keep track of excreted dead cells?

    excreted_inulin: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    excreted_fo: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    excreted_lactose: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    excreted_lactate: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    excreted_glucose: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    excreted_cs: float = field(default=0.0, init=False, metadata={"type": "measurement"})

    absorbed_inulin: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    absorbed_fo: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    absorbed_lactose: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    absorbed_lactate: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    absorbed_glucose: float = field(default=0.0, init=False, metadata={"type": "measurement"})
    absorbed_cs: float = field(default=0.0, init=False, metadata={"type": "measurement"})

    ######################################################################
    # static properties

    @property
    def geometry(self) -> tuple[int, int]:
        return self.GRID_WIDTH, self.GRID_HEIGHT

    ######################################################################
    # dynamic properties

    @property
    def num_bacteria(self):
        return self.num_closts + self.num_bifidos + self.num_bacteroids + self.num_desulfos

    @property
    def mean_bacteroid_energy(self):
        return (
            np.mean(self.bacteroid_energy[self.bacteroid_mask]) if self.num_bacteroids > 0 else 0.0
        )

    @property
    def mean_bacteroid_age(self):
        return np.mean(self.bacteroid_age[self.bacteroid_mask]) if self.num_bacteroids > 0 else 0.0

    @property
    def prop_bacteroid_is_seed(self):
        return (
            np.sum(self.bacteroid_is_seed & self.bacteroid_mask) / self.num_bacteroids
            if self.num_bacteroids > 0
            else 0.0
        )

    @property
    def prop_bacteroid_is_stuck(self):
        return (
            np.sum(self.bacteroid_is_stuck & self.bacteroid_mask) / self.num_bacteroids
            if self.num_bacteroids > 0
            else 0.0
        )

    @property
    def bacteroid_location_centroid(self):
        return (
            np.mean(self.bacteroid_locations[self.bacteroid_mask], axis=0)
            if self.num_bacteroids > 0
            else np.array([0.0, 0.0])
        )

    @property
    def mean_bifido_energy(self):
        return np.mean(self.bifido_energy[self.bifido_mask]) if self.num_bifidos > 0 else 0.0

    @property
    def mean_bifido_age(self):
        return np.mean(self.bifido_age[self.bifido_mask]) if self.num_bifidos > 0 else 0.0

    @property
    def prop_bifido_is_seed(self):
        return (
            np.sum(self.bifido_is_seed & self.bifido_mask) / self.num_bifidos
            if self.num_bifidos > 0
            else 0.0
        )

    @property
    def prop_bifido_is_stuck(self):
        return (
            np.sum(self.bifido_is_stuck & self.bifido_mask) / self.num_bifidos
            if self.num_bifidos > 0
            else 0.0
        )

    @property
    def bifido_location_centroid(self):
        return (
            np.mean(self.bifido_locations[self.bifido_mask], axis=0)
            if self.num_bifidos > 0
            else np.array([0.0, 0.0])
        )

    @property
    def mean_clost_energy(self):
        return np.mean(self.clost_energy[self.clost_mask]) if self.num_closts > 0 else 0.0

    @property
    def mean_clost_age(self):
        return np.mean(self.clost_age[self.clost_mask]) if self.num_closts > 0 else 0.0

    @property
    def prop_clost_is_seed(self):
        return (
            np.sum(self.clost_is_seed & self.clost_mask) / self.num_closts
            if self.num_closts > 0
            else 0.0
        )

    @property
    def prop_clost_is_stuck(self):
        return (
            np.sum(self.clost_is_stuck & self.clost_mask) / self.num_closts
            if self.num_closts > 0
            else 0.0
        )

    @property
    def clost_location_centroid(self):
        return (
            np.mean(self.clost_locations[self.clost_mask], axis=0)
            if self.num_closts > 0
            else np.array([0.0, 0.0])
        )

    @property
    def mean_desulfo_energy(self):
        return np.mean(self.desulfo_energy[self.desulfo_mask]) if self.num_desulfos > 0 else 0.0

    @property
    def mean_desulfo_age(self):
        return np.mean(self.desulfo_age[self.desulfo_mask]) if self.num_desulfos > 0 else 0.0

    @property
    def prop_desulfo_is_seed(self):
        return (
            np.sum(self.desulfo_is_seed & self.desulfo_mask) / self.num_desulfos
            if self.num_desulfos > 0
            else 0.0
        )

    @property
    def prop_desulfo_is_stuck(self):
        return (
            np.sum(self.desulfo_is_stuck & self.desulfo_mask) / self.num_desulfos
            if self.num_desulfos > 0
            else 0.0
        )

    @property
    def desulfo_location_centroid(self):
        return (
            np.mean(self.desulfo_locations[self.desulfo_mask], axis=0)
            if self.num_desulfos > 0
            else np.array([0.0, 0.0])
        )

    @property
    def total_cs(self):
        return np.sum(self.cs)

    @property
    def total_fo(self):
        return np.sum(self.fo)

    @property
    def total_glucose(self):
        return np.sum(self.glucose)

    @property
    def total_inulin(self):
        return np.sum(self.inulin)

    @property
    def total_lactate(self):
        return np.sum(self.lactate)

    @property
    def total_lactose(self):
        return np.sum(self.lactose)

    ######################################################################
    # bifidobacteria

    num_bifidos: int = field(init=False, factory=lambda: 0, metadata={"type": "bookkeeping"})
    bifido_pointer: int = field(init=False, factory=lambda: 0, metadata={"type": "bookkeeping"})

    bifido_mask: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_BIFIDOS, dtype=bool), takes_self=True),
        metadata={"type": "bookkeeping"},
    )
    bifido_locations: np.ndarray = field(
        default=Factory(
            lambda self: np.zeros((self.MAX_BIFIDOS, 2), dtype=np.float32), takes_self=True
        ),
        metadata={"type": "agent"},
    )
    bifido_age: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_BIFIDOS, dtype=np.int64), takes_self=True),
        metadata={"type": "agent"},
    )
    bifido_is_seed: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_BIFIDOS, dtype=np.bool_), takes_self=True),
        metadata={"type": "agent"},
    )
    bifido_is_stuck: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_BIFIDOS, dtype=np.bool_), takes_self=True),
        metadata={"type": "agent"},
    )
    bifido_energy: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_BIFIDOS, dtype=np.float32), takes_self=True),
        metadata={"type": "agent"},
    )

    ######################################################################
    # desulfovibro

    num_desulfos: int = field(init=False, factory=lambda: 0, metadata={"type": "bookkeeping"})
    desulfo_pointer: int = field(init=False, factory=lambda: 0, metadata={"type": "bookkeeping"})

    desulfo_mask: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_DESULFOS, dtype=bool), takes_self=True),
        metadata={"type": "bookkeeping"},
    )
    desulfo_locations: np.ndarray = field(
        default=Factory(
            lambda self: np.zeros((self.MAX_DESULFOS, 2), dtype=np.float32), takes_self=True
        ),
        metadata={"type": "agent"},
    )
    desulfo_age: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_DESULFOS, dtype=np.int64), takes_self=True),
        metadata={"type": "agent"},
    )
    desulfo_is_seed: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_DESULFOS, dtype=np.bool_), takes_self=True),
        metadata={"type": "agent"},
    )
    desulfo_is_stuck: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_DESULFOS, dtype=np.bool_), takes_self=True),
        metadata={"type": "agent"},
    )
    desulfo_energy: np.ndarray = field(
        default=Factory(
            lambda self: np.zeros(self.MAX_DESULFOS, dtype=np.float32), takes_self=True
        ),
        metadata={"type": "agent"},
    )

    ######################################################################
    # clostridia

    num_closts: int = field(init=False, factory=lambda: 0, metadata={"type": "bookkeeping"})
    clost_pointer: int = field(init=False, factory=lambda: 0, metadata={"type": "bookkeeping"})

    clost_mask: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_CLOSTS, dtype=bool), takes_self=True),
        metadata={"type": "bookkeeping"},
    )
    clost_locations: np.ndarray = field(
        default=Factory(
            lambda self: np.zeros((self.MAX_CLOSTS, 2), dtype=np.float32), takes_self=True
        ),
        metadata={"type": "agent"},
    )
    clost_age: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_CLOSTS, dtype=np.int64), takes_self=True),
        metadata={"type": "agent"},
    )
    clost_is_seed: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_CLOSTS, dtype=np.bool_), takes_self=True),
        metadata={"type": "agent"},
    )
    clost_is_stuck: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_CLOSTS, dtype=np.bool_), takes_self=True),
        metadata={"type": "agent"},
    )
    clost_energy: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_CLOSTS, dtype=np.float32), takes_self=True),
        metadata={"type": "agent"},
    )

    ######################################################################
    # bacteroides

    num_bacteroids: int = field(init=False, factory=lambda: 0, metadata={"type": "bookkeeping"})
    bacteroid_pointer: int = field(init=False, factory=lambda: 0, metadata={"type": "bookkeeping"})

    bacteroid_mask: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.MAX_BACTEROIDS, dtype=bool), takes_self=True),
        metadata={"type": "bookkeeping"},
    )
    bacteroid_locations: np.ndarray = field(
        default=Factory(
            lambda self: np.zeros((self.MAX_BACTEROIDS, 2), dtype=np.float32), takes_self=True
        ),
        metadata={"type": "agent"},
    )
    bacteroid_age: np.ndarray = field(
        default=Factory(
            lambda self: np.zeros(self.MAX_BACTEROIDS, dtype=np.int64), takes_self=True
        ),
        metadata={"type": "agent"},
    )
    bacteroid_is_seed: np.ndarray = field(
        default=Factory(
            lambda self: np.zeros(self.MAX_BACTEROIDS, dtype=np.bool_), takes_self=True
        ),
        metadata={"type": "agent"},
    )
    bacteroid_is_stuck: np.ndarray = field(
        default=Factory(
            lambda self: np.zeros(self.MAX_BACTEROIDS, dtype=np.bool_), takes_self=True
        ),
        metadata={"type": "agent"},
    )
    bacteroid_energy: np.ndarray = field(
        default=Factory(
            lambda self: np.zeros(self.MAX_BACTEROIDS, dtype=np.float32), takes_self=True
        ),
        metadata={"type": "agent"},
    )

    ######################################################################
    # patches

    glucose: np.ndarray = field(
        default=Factory(
            lambda self: np.full(self.geometry, 0.0, dtype=np.float32), takes_self=True
        ),
        metadata={"type": "molecule"},
    )
    fo: np.ndarray = field(
        default=Factory(
            lambda self: np.full(self.geometry, 0.0, dtype=np.float32), takes_self=True
        ),
        metadata={"type": "molecule"},
    )
    lactose: np.ndarray = field(
        default=Factory(
            lambda self: np.full(self.geometry, 0.0, dtype=np.float32), takes_self=True
        ),
        metadata={"type": "molecule"},
    )
    lactate: np.ndarray = field(
        default=Factory(
            lambda self: np.full(self.geometry, 0.0, dtype=np.float32), takes_self=True
        ),
        metadata={"type": "molecule"},
    )
    inulin: np.ndarray = field(
        default=Factory(
            lambda self: np.full(self.geometry, 0.0, dtype=np.float32), takes_self=True
        ),
        metadata={"type": "molecule"},
    )
    cs: np.ndarray = field(
        default=Factory(
            lambda self: np.full(self.geometry, 0.0, dtype=np.float32), takes_self=True
        ),
        metadata={"type": "molecule"},
    )
    stuck_chance: np.ndarray = field(
        default=Factory(
            lambda self: np.full(self.geometry, 0.0, dtype=np.float32), takes_self=True
        ),
        metadata={"type": "bookkeeping"},
    )

    ######################################################################
    # Bifidobacteria utility functions

    def create_bifido(
        self,
        *,
        location: Iterable | None = None,
        age: int | None = None,
        energy: float | None = None,
        is_seed: bool | None = None,
        is_stuck: bool | None = None,
    ) -> None:
        """
        Create a Bifidobacterium.

        :param location: location to create the bifidobacterium (optional, random if omitted)
        :param age:
        :param is_stuck:
        :param is_seed:
        :param energy:

        :return:
        """

        # make sure there is space
        if self.bifido_pointer >= self.MAX_BIFIDOS:
            self.compact_bifido_arrays()
            # maybe the array is already compacted:
            if self.bifido_pointer >= self.MAX_BIFIDOS:
                self._expand_bifido_arrays()

        if location is None:
            self.bifido_locations[self.bifido_pointer, :] = np.array(
                self.geometry
            ) * self.rng.random(2)
        else:
            self.bifido_locations[self.bifido_pointer, :] = np.array(location).astype(np.float32)

        self.bifido_age[self.bifido_pointer] = age
        self.bifido_energy[self.bifido_pointer] = energy
        self.bifido_is_seed[self.bifido_pointer] = is_seed
        self.bifido_is_stuck[self.bifido_pointer] = is_stuck

        self.bifido_mask[self.bifido_pointer] = True
        self.num_bifidos += 1
        self.bifido_pointer += 1

    def compact_bifido_arrays(self):
        self.bifido_locations[: self.num_bifidos] = self.bifido_locations[self.bifido_mask]
        self.bifido_age[: self.num_bifidos] = self.bifido_age[self.bifido_mask]
        self.bifido_energy[: self.num_bifidos] = self.bifido_energy[self.bifido_mask]
        self.bifido_is_seed[: self.num_bifidos] = self.bifido_is_seed[self.bifido_mask]
        self.bifido_is_stuck[: self.num_bifidos] = self.bifido_is_stuck[self.bifido_mask]

        self.bifido_mask[: self.num_bifidos] = True
        self.bifido_mask[self.num_bifidos :] = False
        self.bifido_pointer = self.num_bifidos

    def _expand_bifido_arrays(self, new_max_bifidos: int | None = None) -> None:
        old_max_bifidos = self.MAX_BIFIDOS
        if new_max_bifidos is None:
            self.MAX_BIFIDOS *= 2
        else:
            if new_max_bifidos <= self.MAX_BIFIDOS:
                return
            self.MAX_BIFIDOS = new_max_bifidos

        self.bifido_locations = np.pad(
            self.bifido_locations,
            pad_width=np.array(((0, self.MAX_BIFIDOS - old_max_bifidos), (0, 0))),
            mode="constant",
            constant_values=(0, 0),
        )
        self.bifido_age = np.pad(
            self.bifido_age,
            pad_width=np.array((0, self.MAX_BIFIDOS - old_max_bifidos)),
            mode="constant",
            constant_values=0,
        )
        self.bifido_energy = np.pad(
            self.bifido_energy,
            pad_width=np.array((0, self.MAX_BIFIDOS - old_max_bifidos)),
            mode="constant",
            constant_values=0.0,
        )
        self.bifido_is_seed = np.pad(
            self.bifido_is_seed,
            pad_width=np.array((0, self.MAX_BIFIDOS - old_max_bifidos)),
            mode="constant",
            constant_values=False,
        )
        self.bifido_is_stuck = np.pad(
            self.bifido_is_stuck,
            pad_width=np.array((0, self.MAX_BIFIDOS - old_max_bifidos)),
            mode="constant",
            constant_values=False,
        )
        self.bifido_mask = np.pad(
            self.bifido_mask,
            pad_width=np.array((0, self.MAX_BIFIDOS - old_max_bifidos)),
            mode="constant",
            constant_values=False,
        )

    ######################################################################
    # Desulfovibro utility functions

    def create_desulfo(
        self,
        *,
        location: Iterable | None = None,
        age: int | None = None,
        energy: float | None = None,
        is_seed: bool | None = None,
        is_stuck: bool | None = None,
    ) -> None:
        """
        Create a Desulfovibro.

        :param location: location to create the desulfovibro (optional, random if omitted)
        :param age:
        :param is_stuck:
        :param is_seed:
        :param energy:

        :return:
        """

        # make sure there is space
        if self.desulfo_pointer >= self.MAX_DESULFOS:
            self.compact_desulfo_arrays()
            # maybe the array is already compacted:
            if self.desulfo_pointer >= self.MAX_DESULFOS:
                self._expand_desulfo_arrays()

        if location is None:
            self.desulfo_locations[self.desulfo_pointer, :] = np.array(
                self.geometry
            ) * self.rng.random(2)
        else:
            self.desulfo_locations[self.desulfo_pointer, :] = np.array(location).astype(np.float32)

        self.desulfo_age[self.desulfo_pointer] = age
        self.desulfo_energy[self.desulfo_pointer] = energy
        self.desulfo_is_seed[self.desulfo_pointer] = is_seed
        self.desulfo_is_stuck[self.desulfo_pointer] = is_stuck

        self.desulfo_mask[self.desulfo_pointer] = True
        self.num_desulfos += 1
        self.desulfo_pointer += 1

    def compact_desulfo_arrays(self):
        self.desulfo_locations[: self.num_desulfos] = self.desulfo_locations[self.desulfo_mask]
        self.desulfo_age[: self.num_desulfos] = self.desulfo_age[self.desulfo_mask]
        self.desulfo_energy[: self.num_desulfos] = self.desulfo_energy[self.desulfo_mask]
        self.desulfo_is_seed[: self.num_desulfos] = self.desulfo_is_seed[self.desulfo_mask]
        self.desulfo_is_stuck[: self.num_desulfos] = self.desulfo_is_stuck[self.desulfo_mask]

        self.desulfo_mask[: self.num_desulfos] = True
        self.desulfo_mask[self.num_desulfos :] = False
        self.desulfo_pointer = self.num_desulfos

    def _expand_desulfo_arrays(self, new_max_desulfos: int | None = None) -> None:
        old_max_desulfos = self.MAX_DESULFOS
        if new_max_desulfos is None:
            self.MAX_DESULFOS *= 2
        else:
            if new_max_desulfos <= self.MAX_DESULFOS:
                return
            self.MAX_DESULFOS = new_max_desulfos

        self.desulfo_locations = np.pad(
            self.desulfo_locations,
            pad_width=np.array(((0, self.MAX_DESULFOS - old_max_desulfos), (0, 0))),
            mode="constant",
            constant_values=(0, 0),
        )
        self.desulfo_age = np.pad(
            self.desulfo_age,
            pad_width=np.array((0, self.MAX_DESULFOS - old_max_desulfos)),
            mode="constant",
            constant_values=0,
        )
        self.desulfo_energy = np.pad(
            self.desulfo_energy,
            pad_width=np.array((0, self.MAX_DESULFOS - old_max_desulfos)),
            mode="constant",
            constant_values=0.0,
        )
        self.desulfo_is_seed = np.pad(
            self.desulfo_is_seed,
            pad_width=np.array((0, self.MAX_DESULFOS - old_max_desulfos)),
            mode="constant",
            constant_values=False,
        )
        self.desulfo_is_stuck = np.pad(
            self.desulfo_is_stuck,
            pad_width=np.array((0, self.MAX_DESULFOS - old_max_desulfos)),
            mode="constant",
            constant_values=False,
        )
        self.desulfo_mask = np.pad(
            self.desulfo_mask,
            pad_width=np.array((0, self.MAX_DESULFOS - old_max_desulfos)),
            mode="constant",
            constant_values=False,
        )

    ######################################################################
    # Clostridia utility functions

    def create_clost(
        self,
        *,
        location: Iterable | None = None,
        age: int | None = None,
        energy: float | None = None,
        is_seed: bool | None = None,
        is_stuck: bool | None = None,
    ) -> None:
        """
        Create a Clostridium.

        :param location: location to create the clostridium (optional, random if omitted)
        :param age:
        :param is_stuck:
        :param is_seed:
        :param energy:

        :return:
        """

        # make sure there is space
        if self.clost_pointer >= self.MAX_CLOSTS:
            self.compact_clost_arrays()
            # maybe the array is already compacted:
            if self.clost_pointer >= self.MAX_CLOSTS:
                self._expand_clost_arrays()

        if location is None:
            self.clost_locations[self.clost_pointer, :] = np.array(self.geometry) * self.rng.random(
                2
            )
        else:
            self.clost_locations[self.clost_pointer, :] = np.array(location).astype(np.float32)

        self.clost_age[self.clost_pointer] = age
        self.clost_energy[self.clost_pointer] = energy
        self.clost_is_seed[self.clost_pointer] = is_seed
        self.clost_is_stuck[self.clost_pointer] = is_stuck

        self.clost_mask[self.clost_pointer] = True
        self.num_closts += 1
        self.clost_pointer += 1

    def compact_clost_arrays(self):
        self.clost_locations[: self.num_closts] = self.clost_locations[self.clost_mask]
        self.clost_age[: self.num_closts] = self.clost_age[self.clost_mask]
        self.clost_energy[: self.num_closts] = self.clost_energy[self.clost_mask]
        self.clost_is_seed[: self.num_closts] = self.clost_is_seed[self.clost_mask]
        self.clost_is_stuck[: self.num_closts] = self.clost_is_stuck[self.clost_mask]

        self.clost_mask[: self.num_closts] = True
        self.clost_mask[self.num_closts :] = False
        self.clost_pointer = self.num_closts

    def _expand_clost_arrays(self, new_max_closts: int | None = None) -> None:
        old_max_closts = self.MAX_CLOSTS
        if new_max_closts is None:
            self.MAX_CLOSTS *= 2
        else:
            if new_max_closts < self.MAX_CLOSTS:
                return
            self.MAX_CLOSTS = new_max_closts

        self.clost_locations = np.pad(
            self.clost_locations,
            pad_width=np.array(((0, self.MAX_CLOSTS - old_max_closts), (0, 0))),
            mode="constant",
            constant_values=(0, 0),
        )
        self.clost_age = np.pad(
            self.clost_age,
            pad_width=np.array((0, self.MAX_CLOSTS - old_max_closts)),
            mode="constant",
            constant_values=0,
        )
        self.clost_energy = np.pad(
            self.clost_energy,
            pad_width=np.array((0, self.MAX_CLOSTS - old_max_closts)),
            mode="constant",
            constant_values=0.0,
        )
        self.clost_is_seed = np.pad(
            self.clost_is_seed,
            pad_width=np.array((0, self.MAX_CLOSTS - old_max_closts)),
            mode="constant",
            constant_values=False,
        )
        self.clost_is_stuck = np.pad(
            self.clost_is_stuck,
            pad_width=np.array((0, self.MAX_CLOSTS - old_max_closts)),
            mode="constant",
            constant_values=False,
        )
        self.clost_mask = np.pad(
            self.clost_mask,
            pad_width=np.array((0, self.MAX_CLOSTS - old_max_closts)),
            mode="constant",
            constant_values=False,
        )

    ######################################################################
    # Bacteroides utility functions

    def create_bacteroid(
        self,
        *,
        location: Iterable | None = None,
        age: int | None = None,
        energy: float | None = None,
        is_seed: bool | None = None,
        is_stuck: bool | None = None,
    ) -> None:
        """
        Create a Bacteroides.

        :param location: location to create the bacteroides (optional, random if omitted)
        :param age:
        :param is_stuck:
        :param is_seed:
        :param energy:

        :return:
        """

        # make sure there is space
        if self.bacteroid_pointer >= self.MAX_BACTEROIDS:
            self.compact_bacteroid_arrays()
            # maybe the array is already compacted:
            if self.bacteroid_pointer >= self.MAX_BACTEROIDS:
                self._expand_bacteroid_arrays()

        if location is None:
            self.bacteroid_locations[self.bacteroid_pointer, :] = np.array(
                self.geometry
            ) * self.rng.random(2)
        else:
            self.bacteroid_locations[self.bacteroid_pointer, :] = np.array(location).astype(
                np.float32
            )

        self.bacteroid_age[self.bacteroid_pointer] = age
        self.bacteroid_energy[self.bacteroid_pointer] = energy
        self.bacteroid_is_seed[self.bacteroid_pointer] = is_seed
        self.bacteroid_is_stuck[self.bacteroid_pointer] = is_stuck

        self.bacteroid_mask[self.bacteroid_pointer] = True
        self.num_bacteroids += 1
        self.bacteroid_pointer += 1

    def compact_bacteroid_arrays(self):
        self.bacteroid_locations[: self.num_bacteroids] = self.bacteroid_locations[
            self.bacteroid_mask
        ]
        self.bacteroid_age[: self.num_bacteroids] = self.bacteroid_age[self.bacteroid_mask]
        self.bacteroid_energy[: self.num_bacteroids] = self.bacteroid_energy[self.bacteroid_mask]
        self.bacteroid_is_seed[: self.num_bacteroids] = self.bacteroid_is_seed[self.bacteroid_mask]
        self.bacteroid_is_stuck[: self.num_bacteroids] = self.bacteroid_is_stuck[
            self.bacteroid_mask
        ]

        self.bacteroid_mask[: self.num_bacteroids] = True
        self.bacteroid_mask[self.num_bacteroids :] = False
        self.bacteroid_pointer = self.num_bacteroids

    def _expand_bacteroid_arrays(self, new_max_bacteroids: int | None = None) -> None:
        old_max_bacteroids = self.MAX_BACTEROIDS
        if new_max_bacteroids is None:
            self.MAX_BACTEROIDS *= 2
        else:
            if new_max_bacteroids <= old_max_bacteroids:
                return
            self.MAX_BACTEROIDS = new_max_bacteroids

        self.bacteroid_locations = np.pad(
            self.bacteroid_locations,
            pad_width=np.array(((0, self.MAX_BACTEROIDS - old_max_bacteroids), (0, 0))),
            mode="constant",
            constant_values=(0, 0),
        )
        self.bacteroid_age = np.pad(
            self.bacteroid_age,
            pad_width=np.array((0, self.MAX_BACTEROIDS - old_max_bacteroids)),
            mode="constant",
            constant_values=0,
        )
        self.bacteroid_energy = np.pad(
            self.bacteroid_energy,
            pad_width=np.array((0, self.MAX_BACTEROIDS - old_max_bacteroids)),
            mode="constant",
            constant_values=0.0,
        )
        self.bacteroid_is_seed = np.pad(
            self.bacteroid_is_seed,
            pad_width=np.array((0, self.MAX_BACTEROIDS - old_max_bacteroids)),
            mode="constant",
            constant_values=False,
        )
        self.bacteroid_is_stuck = np.pad(
            self.bacteroid_is_stuck,
            pad_width=np.array((0, self.MAX_BACTEROIDS - old_max_bacteroids)),
            mode="constant",
            constant_values=False,
        )
        self.bacteroid_mask = np.pad(
            self.bacteroid_mask,
            pad_width=np.array((0, self.MAX_BACTEROIDS - old_max_bacteroids)),
            mode="constant",
            constant_values=False,
        )

    ######################################################################
    # saving and loading the model state

    def save(self, filename: str, *, write_mode: str = "a"):
        """
        Record the model state to an HDF5 file.
        :param filename:
        :param write_mode: e.g. append, write
        :return:
        """

        with h5py.File(filename, write_mode) as h5file:
            grp: h5py.Group = h5file.create_group(str(self.ticks))
            for fld in fields(type(self)):
                # skip things that can be automatically reconstructed
                if self.field_is_bookkeeping(fld):
                    continue

                value = getattr(self, fld.name)

                if self.field_is_control(fld):
                    ds = grp.create_dataset(fld.name, data=np.void(dill.dumps(value)))
                    ds.attrs["type"] = fld.metadata["type"]
                    ds_val = grp.create_dataset(
                        fld.name + "_value", shape=(), data=value(self.ticks - 1)
                    )
                    ds_val.attrs["type"] = fld.metadata["type"] + "_value"
                elif np.isscalar(value):
                    # scalars can be directly saved
                    ds = grp.create_dataset(fld.name, shape=(), dtype=type(value), data=value)
                    ds.attrs["type"] = fld.metadata["type"]
                else:
                    # numpy arrays may need a bit of filtering first
                    # convert any enum types to ints
                    if np.issubdtype(value.dtype, np.object_):
                        value = value.astype(int)
                    # no need for float64's
                    if np.issubdtype(value.dtype, np.floating):
                        value = value.astype(np.float32)
                    # filter agent arrays to just the meaningful entries
                    if self.field_is_agent(fld):
                        for bact_type in ["bacteroid", "bifido", "clost", "desulfo"]:
                            if fld.name.startswith(bact_type):
                                value = value[getattr(self, f"{bact_type}_mask")]
                                break
                    ds = grp.create_dataset(
                        fld.name,
                        shape=value.shape,
                        dtype=value.dtype,
                        data=value,
                        compression="gzip",
                        compression_opts=9,
                    )
                    ds.attrs["type"] = fld.metadata["type"]
                    if self.field_is_molecule(fld):
                        ds.dims[0].label = "x"
                        ds.dims[1].label = "y"
                    elif self.field_is_agent(fld):
                        ds.dims[0].label = "cell_idx"
                        if fld.name.endswith("_locations"):
                            ds.dims[1].label = "xy"

            # also save computed properties.
            # These won't be reloaded, but it's nice to have them in the hdf5
            field_names = [f.name for f in fields(self.__class__)]
            computed_properties = [
                cp_name
                for cp_name in dir(self)
                if not cp_name.startswith("_")
                and cp_name not in field_names
                and (
                    np.isscalar(getattr(self, cp_name))
                    or isinstance(getattr(self, cp_name), np.ndarray)
                )
                and not isinstance(getattr(self, cp_name), str)
            ]
            for cp_name in computed_properties:
                value = getattr(self, cp_name)
                if np.isscalar(value):
                    ds = grp.create_dataset(cp_name, shape=(), dtype=type(value), data=value)
                else:
                    # no need for float64's
                    if np.issubdtype(value.dtype, np.floating):
                        value = value.astype(np.float32)
                    ds = grp.create_dataset(
                        cp_name, shape=value.shape, dtype=value.dtype, data=value
                    )
                ds.attrs["type"] = "computed_property"

    @classmethod
    def load(cls, filename: str, time: int) -> "GutPython":
        """
        Instantiate the model from an HDF5 file.
        :param filename:
        :param time: which time slice to load from
        :return:
        """

        cell_types = ["bacteroid", "bifido", "clost", "desulfo"]

        cell_fields = [f.name for f in fields(cls) if cls.field_is_agent(f)]
        # parameters, molecular fields, etc.
        non_cell_init_fields = [
            f.name
            for f in fields(cls)
            if cls.field_is_parameter(f)
            or cls.field_is_init_parameter(f)
            or cls.field_is_molecule(f)
        ]
        # measurements
        measurement_fields = [f.name for f in fields(cls) if cls.field_is_measurement(f)]
        # controls
        control_fields = [f.name for f in fields(cls) if cls.field_is_control(f)]

        with h5py.File(filename, "r") as h5file:
            grp = cast(h5py.Group, h5file[str(time)])

            def ds(key: str) -> h5py.Dataset:
                return cast(h5py.Dataset, grp[key])

            controls = {f: dill.loads(ds(f)[()]) for f in control_fields}
            init_fields = {f: cls.fix_scalar_type(ds(f)[()]) for f in non_cell_init_fields}

            # parameters and molecular fields are loaded via init
            model = cls(
                **dict(init_fields, **controls),
            )

            # scalars not initialized by init
            model.ticks = int(ds("ticks")[()])  # Shouldn't this be the same as time?
            for f in measurement_fields:
                setattr(model, f, cls.fix_scalar_type(ds(f)[()]))

            # agents
            # ensure there is enough space for agents
            num_cells = {
                cell_type: ds(f"{cell_type}_locations").shape[0] for cell_type in cell_types
            }
            for cell_type in cell_types:
                getattr(model, f"_expand_{cell_type}_arrays")(num_cells[cell_type])
            # load agent properties
            for f in cell_fields:
                cell_type = f.split("_")[0]
                model_field = getattr(model, f)
                model_field[: num_cells[cell_type]] = ds(f)[()]
            # configure the bookkeeping fields
            for cell_type in cell_types:
                setattr(model, f"num_{cell_type}s", num_cells[cell_type])
                setattr(model, f"{cell_type}_pointer", num_cells[cell_type])
                cell_mask = getattr(model, f"{cell_type}_mask")
                cell_mask[: num_cells[cell_type]] = True
                cell_mask[num_cells[cell_type] :] = False

        # computed bookkeeping field
        model.set_stuck_chance()

        return model

    ######################################################################
    # initialization code

    # def __attrs_post_init__(self):
    #     self.setup()

    def setup(self):
        #   create-bifidos (initNumBifidos * (1 - seedPercent / 100)) [
        #     ;;create non-seeds
        #     set color blue
        #     set size 0.25
        #     set label-color blue - 2
        #     set energy 100
        #     set excrete false
        #     set isSeed false
        #     set isStuck true
        #     set age random 1000
        # 	  set flowConst 1 ;; can use this to edit the breed specific flow distance
        # 	  set doubConst 1
        #     setxy random-xcor random-ycor
        #   ]
        #   create-bifidos (initNumBifidos * (seedPercent / 100)) [
        #     ;;create seeds
        #     set color blue
        #     set size 0.25
        #     set label-color blue - 2
        #     set energy 100
        #     set excrete false
        #     set isSeed true
        #     set isStuck true
        #     set age random 1000
        # 	  set flowConst 1 ;; can use this to edit the breed specific flow distance
        # 	  set doubConst 1
        #     setxy random-xcor random-ycor
        #   ]
        self.num_bifidos = 0
        self.bifido_pointer = 0
        self.bifido_mask[:] = False
        num_seed_bifidos = int(self.init_num_bifidos * self.seed_percent / 100)
        for idx in range(self.init_num_bifidos):
            self.create_bifido(
                energy=100,
                is_seed=idx < num_seed_bifidos,
                is_stuck=True,
                age=int(self.rng.integers(low=0, high=1000)),
            )

        # create-desulfos (initNumDesulfos * (1 - seedPercent / 100)) [
        #     ;;create non-seeds
        #     set color green
        #     set size 0.25
        #     set energy 100
        #     set excrete false
        #     set isSeed false
        #     set isStuck true
        # 	  set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy random-xcor random-ycor
        #   ]
        #
        #   create-desulfos (initNumDesulfos * (seedPercent / 100)) [
        #     ;;create seeds
        #     set color green
        #     set size 0.25
        #     set energy 100
        #     set excrete false
        #     set isSeed true
        #     set isStuck true
        # 	  set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy random-xcor random-ycor
        #   ]
        self.num_desulfos = 0
        self.desulfo_pointer = 0
        self.desulfo_mask[:] = False
        num_seed_desulfos = int(self.init_num_desulfos * self.seed_percent / 100)
        for idx in range(self.init_num_desulfos):
            self.create_desulfo(
                energy=100,
                is_seed=idx < num_seed_desulfos,
                is_stuck=True,
                age=int(self.rng.integers(low=0, high=1000)),
            )

        # create-closts (initNumClosts * (1 - seedPercent / 100)) [
        #     ;;create non-seeds
        #     set color red
        #     set size 0.25
        #     set energy 100
        #     set excrete false
        #     set isSeed false
        #     set isStuck true
        # 	  set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy random-xcor random-ycor
        #   ]
        #
        #     create-closts (initNumClosts * (seedPercent / 100)) [
        #     ;;create seeds
        #     set color red
        #     set size 0.25
        #     set energy 100
        #     set excrete false
        #     set isSeed true
        #     set isStuck true
        # 	  set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy random-xcor random-ycor
        #   ]
        self.num_closts = 0
        self.clost_pointer = 0
        self.clost_mask[:] = False
        num_seed_closts = int(self.init_num_closts * self.seed_percent / 100)
        for idx in range(self.init_num_closts):
            self.create_clost(
                energy=100,
                is_seed=idx < num_seed_closts,
                is_stuck=True,
                age=int(self.rng.integers(low=0, high=1000)),
            )

        # create-bacteroides (initNumBacteroides * (1 - seedPercent / 100)) [
        #     ;;create non-seeds
        #     set color grey
        #     set size 0.25
        #     set energy 100
        #     set excrete false
        #     set isSeed false
        #     set isStuck true
        # 	  set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy random-xcor random-ycor
        #   ]
        #
        #   create-bacteroides (initNumBacteroides * (seedPercent / 100)) [
        #     ;;create seeds
        #     set color grey
        #     set size 0.25
        #     set energy 100
        #     set excrete false
        #     set isSeed true
        #     set isStuck true
        # 	  set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy random-xcor random-ycor
        #   ]
        self.num_bacteroids = 0
        self.bacteroid_pointer = 0
        self.bacteroid_mask[:] = False
        num_seed_bacteroids = int(self.init_num_bacteroids * self.seed_percent / 100)
        for idx in range(self.init_num_bacteroids):
            self.create_bacteroid(
                energy=100,
                is_seed=idx < num_seed_bacteroids,
                is_stuck=True,
                age=int(self.rng.integers(low=0, high=1000)),
            )

        # ;; initializes the patch variables
        #   ask patches [
        #     set glucose 0
        #     set FO 0
        #     set lactose 0
        #     set lactate 0
        #     set inulin 0
        #     set CS 0
        #     set glucosePrev 0
        #     set FOPrev 0
        #     set lactosePrev 0
        #     set lactatePrev 0
        #     set inulinPrev 0
        #     set CSPrev 0
        #     set glucoseReserve 0
        #     set FOReserve 0
        #     set lactoseReserve 0
        #     set lactateReserve 0
        #     set inulinReserve 0
        #     set CSReserve 0
        #     set stuckChance 0
        #   ]

        self.glucose[:, :] = 0.0
        self.fo[:, :] = 0.0
        self.lactose[:, :] = 0.0
        self.lactate[:, :] = 0.0
        self.inulin[:, :] = 0.0
        self.cs[:, :] = 0.0

        self.stuck_chance[:, :] = 0.0

        #   ;; setup the true absorption rate
        #   setTrueAbs

        # ACK: refactored away
        # self.set_true_abs()

        #   ;; set up the stuckChance
        #   setStuckChance

        self.set_stuck_chance()

        #   ;; Setup for stop if negative metas
        #   set negMeta false

        # self.neg_meta = False

        #   ;; set time to zero
        #   reset-ticks

        self.ticks = 0

    def go(self):
        if self.sim_terminated:
            logger.error("Simulation has already terminated! Refusing to continue.")
            return

        # to go
        # ;; This function determines the behavior at each time tick
        #
        #   ;; stop if error or unexpected output
        #   stopCheck

        # ACK: removed
        # self.stop_check()
        if self.num_bacteria > 1000000:
            logger.warning(f"Simulation overwhelmed by bacteria N:{self.num_bacteria}")
            self.sim_terminated = True
        if self.num_bacteria <= 0:
            logger.warning("Bacteria died out!")
            self.sim_terminated = True

        #   ;; Modify the energy level of each turtle and metabolite level of each patch
        #   ask patches [
        #     patchEat
        #     storeMetabolites
        #   ]

        self.patch_eat()
        # ACK: removed store_metabolites, since it only updated the _prev fields which were removed
        # self.store_metabolites()

        #   ;; make meta must be in separate ask, sequential tasks
        #   ask patches[
        #     makeMetabolites
        #   ]

        self.make_metabolites()

        #   ;; agents do their other procedures for this tick
        #   bactTickBehavior

        self.bact_tick_behavior()

        #   ;; set the new stuckChance for the patches
        #   setStuckChance

        self.set_stuck_chance()

        #   ;; change the trueAbsorption
        #   setTrueAbs

        # ACK: refactored out
        # self.set_true_abs()

        #   ;; make agents into seeds
        #   createSeeds

        self.create_seeds()

        #   ;; Probiotics or bacteria in
        #   bactIn

        self.bact_in()

        #   ;; Increment time
        #   tick

        self.ticks += 1

        # end

    def patch_eat(self):
        # to patchEat
        # ;; run this on a ask patches to have them start the turtle eating process
        #   ask turtles-here [
        #     set remAttempts 2 ;; reset the number of attempts
        #     set energy (energy - (100 / 1440)) ;; decrease the energy of the bacteria, currently survive 24 hours no
        #     ;; eat
        #   ]

        self.bacteroid_energy -= 100 / 1440  # 1440 = mins in day
        self.bifido_energy -= 100 / 1440  # 1440 = mins in day
        self.clost_energy -= 100 / 1440  # 1440 = mins in day
        self.desulfo_energy -= 100 / 1440  # 1440 = mins in day

        #   let allMetas (list CS FO glucose inulin lactate lactose);; list containing numbers of all the metas
        #   set avaMetas []
        #
        #   ;; initialize the two lists
        #   let hungryBact (turtles-here with [(energy < 80) and (remAttempts > 0)])
        #   let i 0
        #   while [i < (length(allMetas))][
        #     if (item i allMetas >= 1) [
        #       set avaMetas lput (i + 10) avaMetas
        #     ]
        #     set i (i + 1)
        #   ]
        #   let iter 0 ;; used to limit the number of times the next while loop will occur, arbitrary
        #   ;; do the eating till no metas or not hungry
        #   while [(length(avaMetas) > 0) and any? hungryBact and iter < 100] [
        #     ;; code here to randomly select a turtle from hungryBact and then ask it to run bactEat with a random
        #     ;; meta from ava. list
        #     ask one-of hungryBact [
        #       bactEat(one-of avaMetas)
        #       set remAttempts remAttempts - 1
        #     ]
        #     ;;re-bound agent set
        #     set hungryBact (turtles-here with [(energy < 80) and (remAttempts > 0)])
        #
        #     set iter (iter + 1)
        #   ]
        # end

        hungry_bacteroid_mask = self.bacteroid_mask & (self.bacteroid_energy < 80)
        hungry_bifido_mask = self.bifido_mask & (self.bifido_energy < 80)
        hungry_clost_mask = self.clost_mask & (self.clost_energy < 80)
        hungry_desulfo_mask = self.desulfo_mask & (self.desulfo_energy < 80)

        remaining_attempts = {
            "bacteroid": np.full_like(self.bacteroid_mask, 2, dtype=np.int8),
            "bifido": np.full_like(self.bifido_mask, 2, dtype=np.int8),
            "clost": np.full_like(self.clost_mask, 2, dtype=np.int8),
            "desulfo": np.full_like(self.desulfo_mask, 2, dtype=np.int8),
        }

        hungry_bacteria = list(
            itertools.chain(
                zip(itertools.repeat("bacteroid"), np.where(hungry_bacteroid_mask)[0]),
                zip(itertools.repeat("bifido"), np.where(hungry_bifido_mask)[0]),
                zip(itertools.repeat("clost"), np.where(hungry_clost_mask)[0]),
                zip(itertools.repeat("desulfo"), np.where(hungry_desulfo_mask)[0]),
            )
        )

        eating_iters = 0
        max_eating_iters = 100 * self.GRID_HEIGHT * self.GRID_WIDTH  # 100 per patch
        metabolites = ["cs", "fo", "glucose", "inulin", "lactate", "lactose"]

        while eating_iters < max_eating_iters and len(hungry_bacteria) > 0:
            eating_iters += 1

            hungry_cell_idx = self.rng.integers(low=0, high=len(hungry_bacteria))
            bact_type, idx = hungry_bacteria[hungry_cell_idx]

            metabolite_idx = self.rng.integers(low=0, high=len(metabolites))
            metabolite = metabolites[metabolite_idx]

            self.bact_eat(bact_type, idx, metabolite)

            remaining_attempts[bact_type][idx] -= 1
            if (
                getattr(self, f"{bact_type}_energy")[idx] >= 80
                or remaining_attempts[bact_type][idx] <= 0
            ):
                hungry_bacteria.pop(hungry_cell_idx)

    def bact_eat(self, bact_type, idx, metabolite):

        loc = tuple(np.trunc(getattr(self, f"{bact_type}_locations")[idx]).astype(int))

        match metabolite:
            case "cs":
                #   if (metaNum = 10)[;;CS
                #     ifelse (breed = desulfos)[;; check correct breed
                #       set energy (energy + 50);; increase the energy of the bacteria
                #       ask patch-here [
                #           set CS (CS - 1);; reduce the meta count
                #         if (CS < 1)[;; remove the meta from avaMetas if there is no more of it
                #           set avaMetas remove 10 avaMetas
                #         ]
                #       ]
                #     ]
                #     [;;else
                #       ;;do nothing
                #     ]
                #   ]
                if bact_type != "desulfo":
                    return
                if self.cs[loc] >= 1:
                    getattr(self, f"{bact_type}_energy")[idx] += 50
                    self.cs[loc] -= 1
            case "fo":
                #   if (metaNum = 11)[;;FO
                #     ifelse (breed = closts or breed = bacteroides)[
                #       set energy (energy + 25)
                #       ask patch-here [
                #         set FO (FO - 1)
                #         if (FO < 1)[
                #           set avaMetas remove 11 avaMetas
                #         ]
                #       ]
                #     ]
                #     [;;else
                #       if(breed = bifidos)[
                #         set energy (energy + 50)
                #         ask patch-here [
                #           set FO (FO - 1)
                #           if (FO < 1)[
                #             set avaMetas remove 11 avaMetas
                #           ]
                #         ]
                #         ask patch-here [
                #           set lactate (lactate + bifido-lactate-production)
                #         ]
                #       ]
                #     ];;end else
                #   ]
                if bact_type not in {"clost", "bacteroid", "bifido"}:
                    return
                if self.fo[loc] >= 1:
                    getattr(self, f"{bact_type}_energy")[idx] += 50 if bact_type == "bifido" else 25
                    self.fo[loc] -= 1
                    if bact_type == "bifido":
                        self.lactate[loc] += self.bifido_lactate_production
            case "glucose":
                #   if (metaNum = 12)[;;GLUCOSE
                #     ifelse (breed = closts or breed = bacteroides)[
                #       set energy (energy + 50)
                #       ask patch-here [
                #         set glucose (glucose - 1)
                #         if (glucose < 1)[
                #           set avaMetas remove 12 avaMetas
                #         ]
                #       ]
                #     ]
                #     [;;else
                #       if (breed = bifidos) [
                #         set energy (energy + 25)
                #         ask patch-here [
                #         	set glucose (glucose - 1)
                #           if (glucose < 1)[
                #             set avaMetas remove 12 avaMetas
                #           ]
                #         ]
                #         ask patch-here [
                #           set lactate (lactate + bifido-lactate-production)
                #         ]
                #       ]
                #     ];;end else
                #   ]
                if bact_type not in {"clost", "bacteroid", "bifido"}:
                    return
                if self.glucose[loc] >= 1:
                    getattr(self, f"{bact_type}_energy")[idx] += 25 if bact_type == "bifido" else 50
                    self.glucose[loc] -= 1
                    if bact_type == "bifido":
                        self.lactate[loc] += self.bifido_lactate_production
            case "inulin":
                #   if (metaNum = 13)[;;INULIN
                #     ifelse (breed = closts or breed = bacteroides)[
                #       set energy (energy + 25)
                #       ask patch-here [
                #         set inulin (inulin - 1)
                #         if (inulin < 1)[
                #           set avaMetas remove 13 avaMetas
                #         ]
                #       ]
                #     ]
                #     [;;else
                #       if (breed = bifidos) [
                #       set energy (energy + 25)
                #         ask patch-here [
                #           	set inulin (inulin - 1)
                #           if (inulin < 1)[
                #             set avaMetas remove 13 avaMetas
                #           ]
                #         ]
                #         ask patch-here [
                #           set lactate (lactate + bifido-lactate-production)
                #         ]
                #       ]
                #     ];;end else
                #   ]
                if bact_type not in {"clost", "bacteroid", "bifido"}:
                    return
                if self.inulin[loc] >= 1:
                    getattr(self, f"{bact_type}_energy")[idx] += 25
                    self.inulin[loc] -= 1
                    if bact_type == "bifido":
                        self.lactate[loc] += self.bifido_lactate_production
            case "lactate":
                #   if (metaNum = 14)[;;LACTATE
                #     ifelse (breed = (desulfos))[
                #       set energy (energy + 50)
                #       ask patch-here [
                #         set lactate (lactate - 1)
                #         if (lactate < 1)[
                #           set avaMetas remove 14 avaMetas
                #         ]
                #       ]
                #     ]
                #     [;;else
                #       ;;do nothing
                #     ]
                #   ]
                if bact_type != "desulfo":
                    return
                if self.lactate[loc] >= 1:
                    getattr(self, f"{bact_type}_energy")[idx] += 50
                    self.lactate[loc] -= 1
            case "lactose":
                #   ifelse (metaNum = 15)[;;LACTOSE
                #     ifelse (breed = closts or breed = bacteroides)[
                #       ifelse (breed = closts)[
                #         set energy (energy + 25)
                #       ]
                #       [;;else
                #         set energy (energy + 50)
                #       ];;end else
                #       ask patch-here [
                #         set lactose (lactose - 1)
                #         if (lactose < 1)[
                #           set avaMetas remove 15 avaMetas
                #         ]
                #       ]
                #     ]
                #     [;;else
                #       if (breed = bifidos) [
                #         set energy (energy + 50)
                #         ask patch-here [
                #           	set lactose (lactose - 1)
                #           if (lactose < 1)[
                #             set avaMetas remove 15 avaMetas
                #           ]
                #         ]
                #         ask patch-here [
                #           set lactate (lactate + bifido-lactate-production)
                #         ]
                #       ]
                #     ];;end else
                #   ]
                if bact_type not in {"clost", "bacteroid", "bifido"}:
                    return
                if self.lactose[loc] >= 1:
                    getattr(self, f"{bact_type}_energy")[idx] += 25 if bact_type == "clost" else 50
                    self.lactose[loc] -= 1
                    if bact_type == "bifido":
                        self.lactate[loc] += self.bifido_lactate_production
            case _:
                assert False

    def make_metabolites(self):
        """
        Handle flow and absorption of metabolites.

        :return: None
        """

        total_bacteria = self.num_bacteria
        if total_bacteria <= 0:
            logger.warning("Bacteria died out!")
            self.sim_terminated = True
            do_absorption = False
            true_absorption = 0.0
        elif self.absorption == 0.0:
            do_absorption = False
            true_absorption = 0.0
        else:
            do_absorption = True
            # TODO: package the constants
            true_absorption = self.absorption * (
                self.absorption_constant
                / (
                    (0.8 * (self.num_desulfos / total_bacteria))
                    + (1 * (self.num_closts / total_bacteria))
                    + (1.2 * (self.num_bacteroids / total_bacteria))
                    + (0.7 * (self.num_bifidos / total_bacteria))
                )
            )

        lower_flow_dist: int = math.floor(self.flow_dist)
        upper_flow_dist: int = lower_flow_dist + 1
        frac = self.flow_dist - lower_flow_dist

        for metabolite_name, metabolite, metabolite_inflow in [
            ("inulin", self.inulin, self.inulin_inflow(self.ticks)),
            ("fo", self.fo, self.fo_inflow(self.ticks)),
            ("lactose", self.lactose, self.lactose_inflow(self.ticks)),
            ("lactate", self.lactate, self.lactate_inflow(self.ticks)),
            ("glucose", self.glucose, self.glucose_inflow(self.ticks)),
            ("cs", self.cs, self.cs_inflow(self.ticks)),
        ]:
            # amount per patch. (i.e. evenly distributed vertically, per unit length in horizontal direction)
            metabolite_inflow_amt_per_patch = metabolite_inflow / (
                self.geometry[1] * self.flow_dist
            )

            if frac > 0.0:
                if lower_flow_dist == 0:
                    setattr(
                        self, f"excreted_{metabolite_name}", float(frac * np.sum(metabolite[-1, :]))
                    )
                    metabolite[1:, :] = frac * metabolite[:-1, :] + (1 - frac) * metabolite[1:, :]
                    metabolite[0, :] = (
                        metabolite[0, :] * (1 - frac) + metabolite_inflow_amt_per_patch * frac
                    )
                else:
                    setattr(
                        self,
                        f"excreted_{metabolite_name}",
                        float(
                            np.sum(metabolite[-lower_flow_dist:, :])
                            + frac * np.sum(metabolite[-upper_flow_dist, :])
                        ),
                    )
                    metabolite[upper_flow_dist:, :] = (
                        frac * metabolite[:-upper_flow_dist, :]
                        + (1 - frac) * metabolite[1:-lower_flow_dist, :]
                    )
                    metabolite[lower_flow_dist, :] = (
                        metabolite[0, :] * (1 - frac) + metabolite_inflow_amt_per_patch * frac
                    )
            else:
                setattr(
                    self,
                    f"excreted_{metabolite_name}",
                    float(np.sum(metabolite[-lower_flow_dist:, :])),
                )
                metabolite[lower_flow_dist:, :] = metabolite[:-lower_flow_dist, :]

            if lower_flow_dist > 0:
                metabolite[:lower_flow_dist, :] = metabolite_inflow_amt_per_patch

            np.clip(metabolite, 0, 1000, out=metabolite)
            metabolite[metabolite < 0.001] = 0

            if do_absorption:
                absorption = (
                    true_absorption
                    * (
                        1
                        - self.reserve_fraction
                        * np.linspace(1.0, 0.0, self.GRID_WIDTH)[:, np.newaxis]
                    )
                    * metabolite
                )
                setattr(
                    self,
                    f"absorbed_{metabolite_name}",
                    float(np.sum(absorption[:, :])),
                )
                metabolite[:, :] -= absorption
            else:
                setattr(
                    self,
                    f"absorbed_{metabolite_name}",
                    0.0,
                )

    def bact_tick_behavior(self):
        # to bactTickBehavior
        # ;; reproduce the chosen turtle
        #   ask bifidos [
        #     flowMove ;; movement of the bacteria by flow
        #   ;;randMove ;; movement of the bacteria by a combination of motility and other random forces
        #     checkStuck ;; check if the bacteria becomes stuck or unstuck
        #     deathBifidos ;; check that the energy of the bacteria is enough, otherwise bacteria dies
        #     if (age mod bifidoDoub = 0 and age != 0)[ ;;this line controls on what tick mod reproduce
        #       reproduceBact ;; run the reproduce code for bacteria
        #     ]
        #   	set age (age + 1) ;; increase the age of the bacteria with each tick
        #   ]

        # flowMove
        movable_bifidos = self.bifido_mask & ~self.bifido_is_stuck & ~self.bifido_is_seed
        self.bifido_locations[movable_bifidos, 0] += self.flow_dist * self.bifido_flow_const

        # excrete
        bifido_excrete = (
            self.bifido_locations[:, 0].astype(np.float32) >= self.GRID_WIDTH
        ) & self.bifido_mask
        self.excreted_bifidos = int(np.sum(bifido_excrete))
        self.bifido_mask[bifido_excrete] = False
        self.num_bifidos -= self.excreted_bifidos

        # deathBifidos
        to_kill = self.bifido_mask & (self.bifido_energy <= 0)
        self.bifido_mask[to_kill] = False
        self.num_bifidos -= int(np.sum(to_kill))

        # checkStuck
        bifido_locs = tuple(
            np.minimum(self.bifido_locations.astype(np.int64), np.array(self.geometry) - 1).T
        )
        sticking_bifidos = (
            self.bifido_mask
            & ~self.bifido_is_stuck
            & (self.rng.random(*self.bifido_mask.shape) < (self.stuck_chance[bifido_locs] / 100.0))
        )
        self.bifido_is_stuck[sticking_bifidos] = True
        unsticking_bifidos = (
            self.bifido_mask
            & self.bifido_is_stuck
            & (self.rng.random(self.bifido_is_stuck.shape[0]) < (self.unstuck_chance / 100.0))
        )
        self.bifido_is_stuck[unsticking_bifidos] = False

        # reproduce
        to_reproduce = (
            self.bifido_mask
            & (self.bifido_age % self.bifido_doub == 0)
            & (self.bifido_age > 0)
            & (self.bifido_energy > 50)
        )
        # TODO: why don't we
        #  self.bifido_age[idx] = 0
        #  and make sure that 0 <= age < bifido_doub everywhere
        #  this could simplify to_reproduce's formula:
        #  (self.bifido_age % self.bifido_doub == 0) & (self.bifido_age > 0)
        #  -> (self.bifido_age >= self.bifido_doub)
        #  depends though; is enough time correct or is it the cycle
        if np.any(to_reproduce):
            self.bifido_energy[to_reproduce] /= 2
            # making copies so that we don't have issues if the underlying array needs to compact or resize
            new_cell_energies = self.bifido_energy[to_reproduce].copy()
            new_cell_locs = self.bifido_locations[to_reproduce, :].copy()
            for energy, loc in zip(new_cell_energies, new_cell_locs):
                self.create_bifido(
                    location=loc,
                    energy=energy,
                    is_stuck=False,
                    is_seed=False,
                    age=0,
                )

        # age
        self.bifido_age += 1

        #
        #   ask desulfos [;;controls the behavior for the desulfos bacteria
        #     flowMove
        #   ;;randMove
        #     checkStuck
        #     deathDesulfos
        #     if (age mod desulfoDoub = 0 and age != 0)[
        #       reproduceBact
        #     ]
        #   	set age (age + 1)
        #   ]

        # flowMove
        movable_desulfos = self.desulfo_mask & ~self.desulfo_is_stuck & ~self.desulfo_is_seed
        self.desulfo_locations[movable_desulfos, 0] += self.flow_dist * self.desulfo_flow_const

        # excrete
        desulfo_excrete = (
            self.desulfo_locations[:, 0].astype(np.float32) >= self.GRID_WIDTH
        ) & self.desulfo_mask
        self.excreted_desulfos = int(np.sum(desulfo_excrete))
        self.desulfo_mask[desulfo_excrete] = False
        self.num_desulfos -= self.excreted_desulfos

        # deathDesulfos
        to_kill = self.desulfo_mask & (self.desulfo_energy <= 0)
        self.desulfo_mask[to_kill] = False
        self.num_desulfos -= int(np.sum(to_kill))

        # checkStuck
        desulfo_locs = tuple(
            np.minimum(self.desulfo_locations.astype(np.int64), np.array(self.geometry) - 1).T
        )
        sticking_desulfos = (
            self.desulfo_mask
            & ~self.desulfo_is_stuck
            & (
                self.rng.random(*self.desulfo_is_stuck.shape)
                < (self.stuck_chance[desulfo_locs] / 100.0)
            )
        )
        self.desulfo_is_stuck[sticking_desulfos] = True
        unsticking_desulfos = (
            self.desulfo_mask
            & self.desulfo_is_stuck
            & (self.rng.random(*self.desulfo_is_stuck.shape) < (self.unstuck_chance / 100.0))
        )
        self.desulfo_is_stuck[unsticking_desulfos] = False

        # reproduce
        to_reproduce = (
            self.desulfo_mask
            & (self.desulfo_age % self.desulfo_doub == 0)
            & (self.desulfo_age > 0)
            & (self.desulfo_energy > 50)
        )
        if np.any(to_reproduce):
            self.desulfo_energy[to_reproduce] /= 2
            # making copies so that we don't have issues if the underlying array needs to compact or resize
            new_cell_energies = self.desulfo_energy[to_reproduce].copy()
            new_cell_locs = self.desulfo_locations[to_reproduce, :].copy()
            for energy, loc in zip(new_cell_energies, new_cell_locs):
                self.create_desulfo(
                    location=loc,
                    energy=energy,
                    is_stuck=False,
                    is_seed=False,
                    age=0,
                )

        # age
        self.desulfo_age += 1

        #   ask closts [;;controls the behavior for the closts
        #     flowMove
        #   ;;randMove
        #     checkStuck
        #     deathClosts
        #     if (age mod clostDoub = 0 and age != 0)[
        #       reproduceBact
        #     ]
        #   	set age (age + 1)
        #   ]

        # flowMove
        movable_closts = self.clost_mask & ~self.clost_is_stuck & ~self.clost_is_seed
        self.clost_locations[movable_closts, 0] += self.flow_dist * self.clost_flow_const

        # excrete
        clost_excrete = (
            self.clost_locations[:, 0].astype(np.float32) >= self.GRID_WIDTH
        ) & self.clost_mask
        self.excreted_closts = int(np.sum(clost_excrete))
        self.clost_mask[clost_excrete] = False
        self.num_closts -= self.excreted_closts

        # deathClosts
        to_kill = self.clost_mask & (self.clost_energy <= 0)
        self.clost_mask[to_kill] = False
        self.num_closts -= int(np.sum(to_kill))

        # checkStuck
        clost_locs = tuple(
            np.minimum(self.clost_locations.astype(np.int64), np.array(self.geometry) - 1).T
        )
        sticking_closts = (
            self.clost_mask
            & ~self.clost_is_stuck
            & (self.rng.random(*self.clost_mask.shape) < (self.stuck_chance[clost_locs] / 100.0))
        )
        self.clost_is_stuck[sticking_closts] = True
        unsticking_closts = (
            self.clost_mask
            & self.clost_is_stuck
            & (self.rng.random(*self.clost_mask.shape) < (self.unstuck_chance / 100.0))
        )
        self.clost_is_stuck[unsticking_closts] = False

        # reproduce
        to_reproduce = (
            self.clost_mask
            & (self.clost_age % self.clost_doub == 0)
            & (self.clost_age > 0)
            & (self.clost_energy > 50)
        )
        if np.any(to_reproduce):
            self.clost_energy[to_reproduce] /= 2
            # making copies so that we don't have issues if the underlying array needs to compact or resize
            new_cell_energies = self.clost_energy[to_reproduce].copy()
            new_cell_locs = self.clost_locations[to_reproduce, :].copy()
            for energy, loc in zip(new_cell_energies, new_cell_locs):
                self.create_clost(
                    location=loc,
                    energy=energy,
                    is_stuck=False,
                    is_seed=False,
                    age=0,
                )

        # age
        self.clost_age += 1

        #
        #   ask bacteroides [;;controls the behavior for the bacteroides
        #     flowMove
        #   ;;randMove
        #     checkStuck
        #     deathBacteroides
        #     if (age mod bacteroidDoub = 0 and age != 0)[
        #       reproduceBact
        #     ]
        #   	set age (age + 1)
        #   ]
        #

        # flowMove
        movable_bacteroids = (
            self.bacteroid_mask & ~self.bacteroid_is_stuck & ~self.bacteroid_is_seed
        )
        self.bacteroid_locations[movable_bacteroids, 0] += (
            self.flow_dist * self.bacteroid_flow_const
        )

        # excrete
        bacteroid_excrete = (
            self.bacteroid_locations[:, 0].astype(np.float32) >= self.GRID_WIDTH
        ) & self.bacteroid_mask
        self.excreted_bacteroids = int(np.sum(bacteroid_excrete))
        self.bacteroid_mask[bacteroid_excrete] = False
        self.num_bacteroids -= self.excreted_bacteroids

        # deathBacteroids
        to_kill = self.bacteroid_mask & (self.bacteroid_energy <= 0)
        self.bacteroid_mask[to_kill] = False
        self.num_bacteroids -= int(np.sum(to_kill))

        # checkStuck
        bacteroid_locs = tuple(
            np.minimum(self.bacteroid_locations.astype(np.int64), np.array(self.geometry) - 1).T
        )
        sticking_bacteroids = (
            self.bacteroid_mask
            & ~self.bacteroid_is_stuck
            & (
                self.rng.random(self.bacteroid_is_stuck.shape[0])
                < (self.stuck_chance[bacteroid_locs] / 100.0)
            )
        )
        self.bacteroid_is_stuck[sticking_bacteroids] = True
        unsticking_bacteroids = (
            self.bacteroid_mask
            & self.bacteroid_is_stuck
            & (self.rng.random(self.bacteroid_is_stuck.shape[0]) < (self.unstuck_chance / 100.0))
        )
        self.bacteroid_is_stuck[unsticking_bacteroids] = False

        # reproduce
        to_reproduce = (
            self.bacteroid_mask
            & (self.bacteroid_age % self.bacteroid_doub == 0)
            & (self.bacteroid_age > 0)
            & (self.bacteroid_energy > 50)
        )
        if np.any(to_reproduce):
            self.bacteroid_energy[to_reproduce] /= 2
            # making copies so that we don't have issues if the underlying array needs to compact or resize
            new_cell_energies = self.bacteroid_energy[to_reproduce].copy()
            new_cell_locs = self.bacteroid_locations[to_reproduce, :].copy()
            for energy, loc in zip(new_cell_energies, new_cell_locs):
                self.create_bacteroid(
                    location=loc,
                    energy=energy,
                    is_stuck=False,
                    is_seed=False,
                    age=0,
                )

        # age
        self.bacteroid_age += 1

    def create_seeds(self):
        # to createSeeds
        # ;; controls whether an agent becomes a seed or not
        # ;; first checks if the agent is stuck or not
        #   ask patches[
        #     ask turtles-here[
        #       if (isStuck and (random 100 < seedChance))[
        #         set isSeed true
        #       ]
        #     ]
        #   ]
        # end

        self.bacteroid_is_seed[
            self.bacteroid_is_stuck
            & (self.rng.random(self.bacteroid_mask.shape[0]) < (self.seed_chance / 100.0))
        ] = True

        self.bifido_is_seed[
            self.bifido_is_stuck
            & (self.rng.random(self.bifido_mask.shape[0]) < (self.seed_chance / 100.0))
        ] = True

        self.clost_is_seed[
            self.clost_is_stuck
            & (self.rng.random(self.clost_mask.shape[0]) < (self.seed_chance / 100.0))
        ] = True

        self.desulfo_is_seed[
            self.desulfo_is_stuck
            & (self.rng.random(self.desulfo_mask.shape[0]) < (self.seed_chance / 100.0))
        ] = True

    def bact_in(self):
        # to bactIn
        #   ;; controls when probiotics enter system
        #   if ticks mod tickInflow = 0[
        #     inConc
        #   ]
        # end

        if self.ticks % self.tick_in_flow != 0:
            return

        # to inConc
        # ;; controls the amount of each type of bacteria flowing in to the simulation
        # ;; similar to the code in go, but bacteria are now placed at only in the first column
        #
        #   create-bifidos inConcBifidos [
        #     set color blue
        #     set size 1
        #     set label-color blue - 2
        #     set energy 100
        #     set excrete false
        #     set isSeed false
        #     set isStuck false
        #     set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy min-pxcor - 0.5 random-ycor
        #   ]

        for _ in range(int(self.in_conc_bifidos(self.ticks))):
            self.create_bifido(
                energy=100,
                is_seed=False,
                is_stuck=False,
                age=int(self.rng.integers(low=0, high=1000)),
                location=[0.0, self.rng.random()],
            )

        #   create-desulfos inConcDesulfos [
        #     set color green
        #     set size 1
        #     set energy 100
        #     set excrete false
        #     set isSeed false
        #     set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy min-pxcor - 0.5 random-ycor
        #
        #   ]
        #

        for _ in range(int(self.in_conc_desulfos(self.ticks))):
            self.create_desulfo(
                energy=100,
                is_seed=False,
                is_stuck=False,
                age=int(self.rng.integers(low=0, high=1000)),
                location=[0.0, self.rng.random()],
            )

        #   create-closts inConcClosts [
        #     set color red
        #     set size 1
        #     set energy 100
        #     set excrete false
        #     set isSeed false
        #     set isStuck false
        #     set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy min-pxcor - 0.5 random-ycor
        #
        #   ]

        for _ in range(int(self.in_conc_closts(self.ticks))):
            self.create_clost(
                energy=100,
                is_seed=False,
                is_stuck=False,
                age=int(self.rng.integers(low=0, high=1000)),
                location=[0.0, self.rng.random()],
            )

        #   create-bacteroides inConcBacteroides [
        #     set color grey
        #     set size 1
        #     set energy 100
        #     set excrete false
        #     set isSeed false
        #     set isStuck false
        #     set age random 1000
        # 	  set flowConst 1
        # 	  set doubConst 1
        #     setxy min-pxcor - 0.5 random-ycor
        #
        #   ]
        # end

        for _ in range(int(self.in_conc_bacteroids(self.ticks))):
            self.create_bacteroid(
                energy=100,
                is_seed=False,
                is_stuck=False,
                age=int(self.rng.integers(low=0, high=1000)),
                location=[0.0, self.rng.random()],
            )

    def set_stuck_chance(self):
        occupancy = np.zeros(self.geometry, dtype=np.int64)

        geometry_bounds = np.array(self.geometry)

        bifido_patches = self.bifido_locations[self.bifido_mask, :].astype(np.int64)
        # TODO: vectorize. need to check what happens with repeat locs using
        #  occupancy[tuple(bifido_patches.T)] += 1
        for idx in range(bifido_patches.shape[0]):
            if np.all(bifido_patches[idx] < geometry_bounds):
                occupancy[tuple(bifido_patches[idx])] += 1

        desulfo_patches = self.desulfo_locations[self.desulfo_mask, :].astype(np.int64)
        for idx in range(desulfo_patches.shape[0]):
            if np.all(desulfo_patches[idx] < geometry_bounds):
                occupancy[tuple(desulfo_patches[idx])] += 1

        clost_patches = self.clost_locations[self.clost_mask, :].astype(np.int64)
        for idx in range(clost_patches.shape[0]):
            if np.all(clost_patches[idx] < geometry_bounds):
                occupancy[tuple(clost_patches[idx])] += 1

        bacteroid_patches = self.bacteroid_locations[self.bacteroid_mask, :].astype(np.int64)
        for idx in range(bacteroid_patches.shape[0]):
            if np.all(bacteroid_patches[idx] < geometry_bounds):
                occupancy[tuple(bacteroid_patches[idx])] += 1

        self.stuck_chance[:, :] = self.max_stuck_chance * (
            1 - occupancy / (self.mid_stuck_conc + occupancy)
        )
        self.stuck_chance[self.stuck_chance < self.low_stuck_bound] = 0

    def plot_agents(self, ax: Axes, *, base_zorder: int = -1):
        """
        Plot the agents
        :param ax: Axes on which to plot the agents
        :param base_zorder:
        :return:
        """
        ax.clear()

        # bifidobacteria
        ax.scatter(
            *self.bifido_locations[self.bifido_mask, :].T,
            color="blue",
            marker="o",
            s=1,
            zorder=base_zorder + 1,
        )

        # desulfovibro
        ax.scatter(
            *self.desulfo_locations[self.desulfo_mask, :].T,
            color="green",
            marker="o",
            s=1,
            zorder=base_zorder + 1,
        )

        # clostridia
        ax.scatter(
            *self.clost_locations[self.clost_mask, :].T,
            color="red",
            marker="o",
            s=1,
            zorder=base_zorder + 1,
        )

        # bacteroides
        ax.scatter(
            *self.bacteroid_locations[self.bacteroid_mask, :].T,
            color="grey",
            marker="o",
            s=1,
            zorder=base_zorder + 1,
        )

        ax.set_xlim(0, self.geometry[0])
        ax.set_ylim(0, self.geometry[1])

    def plot_field(self, ax: Axes, *, field_name: str, field_max: float | None = None):
        """
        Plot one of the molecular fields.

        :param ax: The axis to plot upon.
        :param field_name: which field to plot
        :param field_max: maximum value the field is expected to take
        :return:
        """
        ax.clear()
        assert field_name in {
            "glucose",
            "fo",
            "lactose",
            "lactate",
            "inulin",
        }, "Unknown field!"
        field_array = getattr(self, field_name)

        if field_max is not None:
            ax.imshow(
                field_array.T,
                vmin=0,
                vmax=field_max,
                origin="lower",
                extent=(0.0, field_array.shape[0], 0.0, field_array.shape[1]),
            )
        else:
            ax.imshow(
                field_array.T,
                vmin=0,
                origin="lower",
                extent=(0.0, field_array.shape[0], 0.0, field_array.shape[1]),
            )

    @staticmethod
    def field_is_bookkeeping(f):
        return (
            hasattr(f, "metadata") and "type" in f.metadata and f.metadata["type"] == "bookkeeping"
        )

    @staticmethod
    def field_is_agent(f):
        return hasattr(f, "metadata") and "type" in f.metadata and f.metadata["type"] == "agent"

    @staticmethod
    def field_is_parameter(f):
        return hasattr(f, "metadata") and "type" in f.metadata and f.metadata["type"] == "parameter"

    @staticmethod
    def field_is_init_parameter(f):
        return (
            hasattr(f, "metadata")
            and "type" in f.metadata
            and f.metadata["type"] == "init_parameter"
        )

    @staticmethod
    def field_is_molecule(f):
        return hasattr(f, "metadata") and "type" in f.metadata and f.metadata["type"] == "molecule"

    @staticmethod
    def field_is_measurement(f):
        return (
            hasattr(f, "metadata") and "type" in f.metadata and f.metadata["type"] == "measurement"
        )

    @staticmethod
    def field_is_control(f):
        return hasattr(f, "metadata") and "type" in f.metadata and f.metadata["type"] == "control"

    @staticmethod
    def fix_scalar_type(f):
        if isinstance(f, np.ndarray):
            if np.issubdtype(f.dtype, np.floating):
                return f.astype(np.float32)
            elif np.issubdtype(f.dtype, np.integer):
                return f.astype(np.int64)
            else:
                return f
        elif np.issubdtype(f, np.floating):
            return float(f)
        elif np.issubdtype(f, np.integer):
            return int(f)
        else:
            return f


if __name__ == "__main__":

    gp = GutPython()
    gp.setup()
    for _ in range(10_000):
        gp.go()
