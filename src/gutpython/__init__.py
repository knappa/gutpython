import itertools
import math
from typing import Final, Iterable, Optional, Tuple

import h5py
import numpy as np
from attr import define, field, fields

BIG_NUM = 3000
MEDIUM_NUM = 200
VERBOSE = False


@define(kw_only=True)
class GutPython:
    GRID_WIDTH: int = field(default=100)
    GRID_HEIGHT: int = field(default=1)
    MAX_BIFIDOS: int = field(default=BIG_NUM)
    MAX_DESULFOS: int = field(default=MEDIUM_NUM)
    MAX_CLOSTS: int = field(default=MEDIUM_NUM)
    MAX_BACTEROIDS: int = field(default=MEDIUM_NUM)
    HARD_BOUND: bool = field(default=True)

    # globals
    #   [ trueAbsorption negMeta testState result
    #   ]

    ######################################################################
    # parameters from interface

    max_stuck_chance: float = field(default=50)  # TODO: understand units. percent?
    low_stuck_bound: float = field(default=2)  # TODO: understand units. percent?
    unstuck_chance: float = field(default=10)  # TODO: understand units. percent?
    mid_stuck_conc: float = field(default=10.0)  # TODO: understand units. percent?
    seed_chance: float = field(default=5.0)  # TODO: understand units. percent?
    seed_percent: float = field(default=5.0)

    absorption: float = field(default=0.0)  # TODO: understand units.
    reserve_fraction: float = field(default=0.0)  # TODO: understand units.

    init_num_bifidos: int = field(default=23562)  # TODO: uhh? Seems awfully specific.
    init_num_bacteroids: int = field(default=5490)  # TODO: uhh? Seems awfully specific.
    init_num_closts: int = field(default=921)  # TODO: uhh? Seems awfully specific.
    init_num_desulfos: int = field(default=70)

    bifido_lactate_production: float = field(default=0.005)

    flow_dist: float = field(default=0.28)

    inulin_inflow: float = field(default=10.0)
    fo_inflow: float = field(default=25.0)
    lactose_inflow: float = field(default=15.0)
    lactate_inflow: float = field(default=0.0)
    glucose_inflow: float = field(default=30.0)
    cs_inflow: float = field(default=0.1)

    bifido_doub: int = field(default=330)
    desulfo_doub: int = field(default=330)
    bacteroid_doub: int = field(default=330)
    clost_doub: int = field(default=330)

    tick_in_flow: int = field(default=1)  # TODO: value?
    in_conc_bacteroids: int = field(default=0)
    in_conc_bifidos: int = field(default=0)
    in_conc_closts: int = field(default=0)
    in_conc_desulfos: int = field(default=0)

    ######################################################################
    # hidden parameters (magic constants)

    absorption_constant: Final[float] = 0.723823204

    ######################################################################
    # other globals

    true_absorption: float = field(default=0.0)  # TODO: understand units.

    # neg_meta: bool = False
    test_state: int = field(default=0)
    ticks: int = field(default=0)

    ######################################################################
    # static properties

    @property
    def geometry(self) -> Tuple[int, int]:
        return self.GRID_WIDTH, self.GRID_HEIGHT

    ######################################################################
    # dynamic properties

    @property
    def num_bacteria(self):
        return self.num_closts + self.num_bifidos + self.num_bacteroids + self.num_bacteroids

    ######################################################################
    # bifidobacteria

    bifido_doub_const: float = field(default=1.0)
    bifido_flow_const: float = field(default=1.0)

    num_bifidos: int = field(init=False, factory=lambda: 0)
    bifido_pointer: int = field(init=False, factory=lambda: 0)

    bifido_mask = field(type=np.ndarray)

    @bifido_mask.default
    def _bifido_mask_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=bool)

    bifido_locations = field(type=np.ndarray)

    @bifido_locations.default
    def _bifido_locations_factory(self):
        return np.zeros((self.MAX_BIFIDOS, 2), dtype=np.float64)

    bifido_dirs = field(type=np.ndarray)

    @bifido_dirs.default
    def _bifido_dirs_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=np.float64)

    bifido_age = field(type=np.ndarray)

    @bifido_age.default
    def _bifido_age_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=np.int64)

    bifido_is_seed = field(type=np.ndarray)

    @bifido_is_seed.default
    def _bifido_is_seed_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=np.bool_)

    bifido_is_stuck = field(type=np.ndarray)

    @bifido_is_stuck.default
    def _bifido_is_stuck_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=np.bool_)

    bifido_remaining_attempts = field(type=np.ndarray)

    @bifido_remaining_attempts.default
    def _bifido_remaining_attempts_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=np.int64)

    bifido_energy = field(type=np.ndarray)

    @bifido_energy.default
    def _bifido_energy_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=np.float64)

    ######################################################################
    # desulfovibro

    desulfo_doub_const: float = field(default=1.0)
    desulfo_flow_const: float = field(default=1.0)

    num_desulfos: int = field(init=False, factory=lambda: 0)
    desulfo_pointer: int = field(init=False, factory=lambda: 0)

    desulfo_mask = field(type=np.ndarray)

    @desulfo_mask.default
    def _desulfo_mask_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=bool)

    desulfo_locations = field(type=np.ndarray)

    @desulfo_locations.default
    def _desulfo_locations_factory(self):
        return np.zeros((self.MAX_DESULFOS, 2), dtype=np.float64)

    desulfo_dirs = field(type=np.ndarray)

    @desulfo_dirs.default
    def _desulfo_dirs_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=np.float64)

    desulfo_age = field(type=np.ndarray)

    @desulfo_age.default
    def _desulfo_age_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=np.int64)

    desulfo_is_seed = field(type=np.ndarray)

    @desulfo_is_seed.default
    def _desulfo_is_seed_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=np.bool_)

    desulfo_is_stuck = field(type=np.ndarray)

    @desulfo_is_stuck.default
    def _desulfo_is_stuck_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=np.bool_)

    desulfo_remaining_attempts = field(type=np.ndarray)

    @desulfo_remaining_attempts.default
    def _desulfo_remaining_attempts_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=np.int64)

    desulfo_energy = field(type=np.ndarray)

    @desulfo_energy.default
    def _desulfo_energy_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=np.float64)

    ######################################################################
    # clostridia

    clost_doub_const: float = field(default=1.0)
    clost_flow_const: float = field(default=1.0)

    num_closts: int = field(init=False, factory=lambda: 0)
    clost_pointer: int = field(init=False, factory=lambda: 0)

    clost_mask = field(type=np.ndarray)

    @clost_mask.default
    def _clost_mask_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=bool)

    clost_locations = field(type=np.ndarray)

    @clost_locations.default
    def _clost_locations_factory(self):
        return np.zeros((self.MAX_CLOSTS, 2), dtype=np.float64)

    clost_dirs = field(type=np.ndarray)

    @clost_dirs.default
    def _clost_dirs_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=np.float64)

    clost_age = field(type=np.ndarray)

    @clost_age.default
    def _clost_age_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=np.int64)

    clost_is_seed = field(type=np.ndarray)

    @clost_is_seed.default
    def _clost_is_seed_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=np.bool_)

    clost_is_stuck = field(type=np.ndarray)

    @clost_is_stuck.default
    def _clost_is_stuck_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=np.bool_)

    clost_remaining_attempts = field(type=np.ndarray)

    @clost_remaining_attempts.default
    def _clost_remaining_attempts_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=np.int64)

    clost_energy = field(type=np.ndarray)

    @clost_energy.default
    def _clost_energy_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=np.float64)

    ######################################################################
    # bacteroides

    bacteroid_doub_const: float = field(default=1.0)
    bacteroid_flow_const: float = field(default=1.0)

    num_bacteroids: int = field(init=False, factory=lambda: 0)
    bacteroid_pointer: int = field(init=False, factory=lambda: 0)

    bacteroid_mask = field(type=np.ndarray)

    @bacteroid_mask.default
    def _bacteroid_mask_factory(self):
        return np.zeros(self.MAX_BACTEROIDS, dtype=bool)

    bacteroid_locations = field(type=np.ndarray)

    @bacteroid_locations.default
    def _bacteroid_locations_factory(self):
        return np.zeros((self.MAX_BACTEROIDS, 2), dtype=np.float64)

    bacteroid_dirs = field(type=np.ndarray)

    @bacteroid_dirs.default
    def _bacteroid_dirs_factory(self):
        return np.zeros(self.MAX_BACTEROIDS, dtype=np.float64)

    bacteroid_age = field(type=np.ndarray)

    @bacteroid_age.default
    def _bacteroid_age_factory(self):
        return np.zeros(self.MAX_BACTEROIDS, dtype=np.int64)

    bacteroid_is_seed = field(type=np.ndarray)

    @bacteroid_is_seed.default
    def _bacteroid_is_seed_factory(self):
        return np.zeros(self.MAX_BACTEROIDS, dtype=np.bool_)

    bacteroid_is_stuck = field(type=np.ndarray)

    @bacteroid_is_stuck.default
    def _bacteroid_is_stuck_factory(self):
        return np.zeros(self.MAX_BACTEROIDS, dtype=np.bool_)

    bacteroid_remaining_attempts = field(type=np.ndarray)

    @bacteroid_remaining_attempts.default
    def _bacteroid_remaining_attempts_factory(self):
        return np.zeros(self.MAX_BACTEROIDS, dtype=np.int64)

    bacteroid_energy = field(type=np.ndarray)

    @bacteroid_energy.default
    def _bacteroid_energy_factory(self):
        return np.zeros(self.MAX_BACTEROIDS, dtype=np.float64)

    ######################################################################
    # patches

    glucose = field(type=np.ndarray)

    @glucose.default
    def _glucose_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    glucose_prev = field(type=np.ndarray)

    @glucose_prev.default
    def _glucose_prev_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    glucose_reserve = field(type=np.ndarray)

    @glucose_reserve.default
    def _glucose_reserve_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    ########################################

    fo = field(type=np.ndarray)

    @fo.default
    def _fo_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    fo_prev = field(type=np.ndarray)

    @fo_prev.default
    def _fo_prev_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    fo_reserve = field(type=np.ndarray)

    @fo_reserve.default
    def _fo_reserve_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    ########################################

    lactose = field(type=np.ndarray)

    @lactose.default
    def _lactose_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    lactose_prev = field(type=np.ndarray)

    @lactose_prev.default
    def _lactose_prev_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    lactose_reserve = field(type=np.ndarray)

    @lactose_reserve.default
    def _lactose_reserve_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    ########################################

    lactate = field(type=np.ndarray)

    @lactate.default
    def _lactate_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    lactate_prev = field(type=np.ndarray)

    @lactate_prev.default
    def _lactate_prev_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    lactate_reserve = field(type=np.ndarray)

    @lactate_reserve.default
    def _lactate_reserve_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    ########################################

    inulin = field(type=np.ndarray)

    @inulin.default
    def _inulin_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    inulin_prev = field(type=np.ndarray)

    @inulin_prev.default
    def _inulin_prev_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    inulin_reserve = field(type=np.ndarray)

    @inulin_reserve.default
    def _inulin_reserve_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    ########################################

    cs = field(type=np.ndarray)

    @cs.default
    def _cs_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    cs_prev = field(type=np.ndarray)

    @cs_prev.default
    def _cs_prev_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    cs_reserve = field(type=np.ndarray)

    @cs_reserve.default
    def _cs_reserve_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    ########################################

    stuck_chance = field(type=np.ndarray)

    @stuck_chance.default
    def _stuck_chance_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

    ######################################################################
    # Bifidobacteria utility functions

    def create_bifido(
        self,
        *,
        location: Optional[Iterable] = None,
        theta: Optional[float] = None,
        age: Optional[int] = None,
        energy: Optional[float] = None,
        is_seed: Optional[bool] = None,
        is_stuck: Optional[bool] = None,
        rem_attempt: int = 0,
    ) -> None:
        """
        Create a Bifidobacterium.

        :param location: location to create the bifidobacterium (optional, random if omitted)
        :param theta: direction of bifidobacterium movement in radians (optional, random if omitted)
        :param age:
        :param rem_attempt:
        :param is_stuck:
        :param is_seed:
        :param energy:

        :return:
        """
        if self.num_bifidos >= self.GRID_WIDTH * self.GRID_HEIGHT:
            if VERBOSE:
                print("Refusing to create a bifidobacterium when there is no room for one.")
            return

        # make sure there is space
        if self.bifido_pointer >= self.MAX_BIFIDOS:
            self.compact_bifido_arrays()
            # maybe the array is already compacted:
            if self.bifido_pointer >= self.MAX_BIFIDOS:
                self._expand_bifido_arrays()

        if location is None:
            self.bifido_locations[self.bifido_pointer, :] = np.array(
                self.geometry
            ) * np.random.rand(2)
        else:
            self.bifido_locations[self.bifido_pointer, :] = np.array(location).astype(np.float64)

        if theta is None:
            theta = 2 * np.pi * np.random.rand() - np.pi
        else:
            theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        self.bifido_dirs[self.bifido_pointer] = theta

        self.bifido_age[self.bifido_pointer] = age
        self.bifido_energy[self.bifido_pointer] = energy
        self.bifido_is_seed[self.bifido_pointer] = is_seed
        self.bifido_is_stuck[self.bifido_pointer] = is_stuck
        self.bifido_remaining_attempts[self.bifido_pointer] = rem_attempt

        self.bifido_mask[self.bifido_pointer] = True
        self.num_bifidos += 1
        self.bifido_pointer += 1

    def compact_bifido_arrays(self):
        self.bifido_locations[: self.num_bifidos] = self.bifido_locations[self.bifido_mask]
        self.bifido_dirs[: self.num_bifidos] = self.bifido_dirs[self.bifido_mask]
        self.bifido_age[: self.num_bifidos] = self.bifido_age[self.bifido_mask]
        self.bifido_energy[: self.num_bifidos] = self.bifido_energy[self.bifido_mask]
        self.bifido_is_seed[: self.num_bifidos] = self.bifido_is_seed[self.bifido_mask]
        self.bifido_is_stuck[: self.num_bifidos] = self.bifido_is_stuck[self.bifido_mask]
        self.bifido_remaining_attempts[: self.num_bifidos] = self.bifido_remaining_attempts[
            self.bifido_mask
        ]

        self.bifido_mask[: self.num_bifidos] = True
        self.bifido_mask[self.num_bifidos :] = False
        self.bifido_pointer = self.num_bifidos

    def _expand_bifido_arrays(self):
        old_max_bifidos = self.MAX_BIFIDOS
        self.MAX_BIFIDOS *= 2

        self.bifido_locations = np.pad(
            self.bifido_locations,
            pad_width=np.array(((0, old_max_bifidos), (0, 0))),
            mode="constant",
            constant_values=(0, 0),
        )
        self.bifido_dirs = np.pad(
            self.bifido_dirs,
            pad_width=np.array((0, old_max_bifidos)),
            mode="constant",
            constant_values=0.0,
        )
        self.bifido_age = np.pad(
            self.bifido_age,
            pad_width=np.array((0, old_max_bifidos)),
            mode="constant",
            constant_values=0,
        )
        self.bifido_energy = np.pad(
            self.bifido_energy,
            pad_width=np.array((0, old_max_bifidos)),
            mode="constant",
            constant_values=0.0,
        )
        self.bifido_is_seed = np.pad(
            self.bifido_is_seed,
            pad_width=np.array((0, old_max_bifidos)),
            mode="constant",
            constant_values=False,
        )
        self.bifido_is_stuck = np.pad(
            self.bifido_is_stuck,
            pad_width=np.array((0, old_max_bifidos)),
            mode="constant",
            constant_values=False,
        )
        self.bifido_remaining_attempts = np.pad(
            self.bifido_remaining_attempts,
            pad_width=np.array((0, old_max_bifidos)),
            mode="constant",
            constant_values=0,
        )
        self.bifido_mask = np.pad(
            self.bifido_mask,
            pad_width=np.array((0, old_max_bifidos)),
            mode="constant",
            constant_values=False,
        )

    ######################################################################
    # Desulfovibro utility functions

    def create_desulfo(
        self,
        *,
        location: Optional[Iterable] = None,
        theta: Optional[float] = None,
        age: Optional[int] = None,
        energy: Optional[float] = None,
        is_seed: Optional[bool] = None,
        is_stuck: Optional[bool] = None,
        rem_attempt: int = 0,
    ) -> None:
        """
        Create a Desulfovibro.

        :param location: location to create the desulfovibro (optional, random if omitted)
        :param theta: direction of desulfovibro movement in radians (optional, random if omitted)
        :param age:
        :param rem_attempt:
        :param is_stuck:
        :param is_seed:
        :param energy:

        :return:
        """
        if self.num_desulfos >= self.GRID_WIDTH * self.GRID_HEIGHT:
            if VERBOSE:
                print("Refusing to create a desulfovibro when there is no room for one.")
            return

        # make sure there is space
        if self.desulfo_pointer >= self.MAX_DESULFOS:
            self.compact_desulfo_arrays()
            # maybe the array is already compacted:
            if self.desulfo_pointer >= self.MAX_DESULFOS:
                self._expand_desulfo_arrays()

        if location is None:
            self.desulfo_locations[self.desulfo_pointer, :] = np.array(
                self.geometry
            ) * np.random.rand(2)
        else:
            self.desulfo_locations[self.desulfo_pointer, :] = np.array(location).astype(np.float64)

        if theta is None:
            theta = 2 * np.pi * np.random.rand() - np.pi
        else:
            theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        self.desulfo_dirs[self.desulfo_pointer] = theta

        self.desulfo_age[self.desulfo_pointer] = age
        self.desulfo_energy[self.desulfo_pointer] = energy
        self.desulfo_is_seed[self.desulfo_pointer] = is_seed
        self.desulfo_is_stuck[self.desulfo_pointer] = is_stuck
        self.desulfo_remaining_attempts[self.desulfo_pointer] = rem_attempt

        self.desulfo_mask[self.desulfo_pointer] = True
        self.num_desulfos += 1
        self.desulfo_pointer += 1

    def compact_desulfo_arrays(self):
        self.desulfo_locations[: self.num_desulfos] = self.desulfo_locations[self.desulfo_mask]
        self.desulfo_dirs[: self.num_desulfos] = self.desulfo_dirs[self.desulfo_mask]
        self.desulfo_age[: self.num_desulfos] = self.desulfo_age[self.desulfo_mask]
        self.desulfo_energy[: self.num_desulfos] = self.desulfo_energy[self.desulfo_mask]
        self.desulfo_is_seed[: self.num_desulfos] = self.desulfo_is_seed[self.desulfo_mask]
        self.desulfo_is_stuck[: self.num_desulfos] = self.desulfo_is_stuck[self.desulfo_mask]
        self.desulfo_remaining_attempts[: self.num_desulfos] = self.desulfo_remaining_attempts[
            self.desulfo_mask
        ]

        self.desulfo_mask[: self.num_desulfos] = True
        self.desulfo_mask[self.num_desulfos :] = False
        self.desulfo_pointer = self.num_desulfos

    def _expand_desulfo_arrays(self):
        old_max_desulfos = self.MAX_DESULFOS
        self.MAX_DESULFOS *= 2

        self.desulfo_locations = np.pad(
            self.desulfo_locations,
            pad_width=np.array(((0, old_max_desulfos), (0, 0))),
            mode="constant",
            constant_values=(0, 0),
        )
        self.desulfo_dirs = np.pad(
            self.desulfo_dirs,
            pad_width=np.array((0, old_max_desulfos)),
            mode="constant",
            constant_values=0.0,
        )
        self.desulfo_age = np.pad(
            self.desulfo_age,
            pad_width=np.array((0, old_max_desulfos)),
            mode="constant",
            constant_values=0,
        )
        self.desulfo_energy = np.pad(
            self.desulfo_energy,
            pad_width=np.array((0, old_max_desulfos)),
            mode="constant",
            constant_values=0.0,
        )
        self.desulfo_is_seed = np.pad(
            self.desulfo_is_seed,
            pad_width=np.array((0, old_max_desulfos)),
            mode="constant",
            constant_values=False,
        )
        self.desulfo_is_stuck = np.pad(
            self.desulfo_is_stuck,
            pad_width=np.array((0, old_max_desulfos)),
            mode="constant",
            constant_values=False,
        )
        self.desulfo_remaining_attempts = np.pad(
            self.desulfo_remaining_attempts,
            pad_width=np.array((0, old_max_desulfos)),
            mode="constant",
            constant_values=0,
        )
        self.desulfo_mask = np.pad(
            self.desulfo_mask,
            pad_width=np.array((0, old_max_desulfos)),
            mode="constant",
            constant_values=False,
        )

    ######################################################################
    # Clostridia utility functions

    def create_clost(
        self,
        *,
        location: Optional[Iterable] = None,
        theta: Optional[float] = None,
        age: Optional[int] = None,
        energy: Optional[float] = None,
        is_seed: Optional[bool] = None,
        is_stuck: Optional[bool] = None,
        rem_attempt: int = 0,
    ) -> None:
        """
        Create a Clostridium.

        :param location: location to create the clostridium (optional, random if omitted)
        :param theta: direction of clostridium movement in radians (optional, random if omitted)
        :param age:
        :param rem_attempt:
        :param is_stuck:
        :param is_seed:
        :param energy:

        :return:
        """
        if self.num_closts >= self.GRID_WIDTH * self.GRID_HEIGHT:
            if VERBOSE:
                print("Refusing to create a clostridium when there is no room for one.")
            return

        # make sure there is space
        if self.clost_pointer >= self.MAX_CLOSTS:
            self.compact_clost_arrays()
            # maybe the array is already compacted:
            if self.clost_pointer >= self.MAX_CLOSTS:
                self._expand_clost_arrays()

        if location is None:
            self.clost_locations[self.clost_pointer, :] = np.array(self.geometry) * np.random.rand(
                2
            )
        else:
            self.clost_locations[self.clost_pointer, :] = np.array(location).astype(np.float64)

        if theta is None:
            theta = 2 * np.pi * np.random.rand() - np.pi
        else:
            theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        self.clost_dirs[self.clost_pointer] = theta

        self.clost_age[self.clost_pointer] = age
        self.clost_energy[self.clost_pointer] = energy
        self.clost_is_seed[self.clost_pointer] = is_seed
        self.clost_is_stuck[self.clost_pointer] = is_stuck
        self.clost_remaining_attempts[self.clost_pointer] = rem_attempt

        self.clost_mask[self.clost_pointer] = True
        self.num_closts += 1
        self.clost_pointer += 1

    def compact_clost_arrays(self):
        self.clost_locations[: self.num_closts] = self.clost_locations[self.clost_mask]
        self.clost_dirs[: self.num_closts] = self.clost_dirs[self.clost_mask]
        self.clost_age[: self.num_closts] = self.clost_age[self.clost_mask]
        self.clost_energy[: self.num_closts] = self.clost_energy[self.clost_mask]
        self.clost_is_seed[: self.num_closts] = self.clost_is_seed[self.clost_mask]
        self.clost_is_stuck[: self.num_closts] = self.clost_is_stuck[self.clost_mask]
        self.clost_remaining_attempts[: self.num_closts] = self.clost_remaining_attempts[
            self.clost_mask
        ]

        self.clost_mask[: self.num_closts] = True
        self.clost_mask[self.num_closts :] = False
        self.clost_pointer = self.num_closts

    def _expand_clost_arrays(self):
        old_max_closts = self.MAX_CLOSTS
        self.MAX_CLOSTS *= 2

        self.clost_locations = np.pad(
            self.clost_locations,
            pad_width=np.array(((0, old_max_closts), (0, 0))),
            mode="constant",
            constant_values=(0, 0),
        )
        self.clost_dirs = np.pad(
            self.clost_dirs,
            pad_width=np.array((0, old_max_closts)),
            mode="constant",
            constant_values=0.0,
        )
        self.clost_age = np.pad(
            self.clost_age,
            pad_width=np.array((0, old_max_closts)),
            mode="constant",
            constant_values=0,
        )
        self.clost_energy = np.pad(
            self.clost_energy,
            pad_width=np.array((0, old_max_closts)),
            mode="constant",
            constant_values=0.0,
        )
        self.clost_is_seed = np.pad(
            self.clost_is_seed,
            pad_width=np.array((0, old_max_closts)),
            mode="constant",
            constant_values=False,
        )
        self.clost_is_stuck = np.pad(
            self.clost_is_stuck,
            pad_width=np.array((0, old_max_closts)),
            mode="constant",
            constant_values=False,
        )
        self.clost_remaining_attempts = np.pad(
            self.clost_remaining_attempts,
            pad_width=np.array((0, old_max_closts)),
            mode="constant",
            constant_values=0,
        )
        self.clost_mask = np.pad(
            self.clost_mask,
            pad_width=np.array((0, old_max_closts)),
            mode="constant",
            constant_values=False,
        )

    ######################################################################
    # Bacteroides utility functions

    def create_bacteroid(
        self,
        *,
        location: Optional[Iterable] = None,
        theta: Optional[float] = None,
        age: Optional[int] = None,
        energy: Optional[float] = None,
        is_seed: Optional[bool] = None,
        is_stuck: Optional[bool] = None,
        rem_attempt: int = 0,
    ) -> None:
        """
        Create a Bacteroides.

        :param location: location to create the bacteroides (optional, random if omitted)
        :param theta: direction of bacteroides movement in radians (optional, random if omitted)
        :param age:
        :param rem_attempt:
        :param is_stuck:
        :param is_seed:
        :param energy:

        :return:
        """
        if self.num_bacteroids >= self.GRID_WIDTH * self.GRID_HEIGHT:
            if VERBOSE:
                print("Refusing to create a bacteroides when there is no room for one.")
            return

        # make sure there is space
        if self.bacteroid_pointer >= self.MAX_BACTEROIDS:
            self.compact_bacteroid_arrays()
            # maybe the array is already compacted:
            if self.bacteroid_pointer >= self.MAX_BACTEROIDS:
                self._expand_bacteroid_arrays()

        if location is None:
            self.bacteroid_locations[self.bacteroid_pointer, :] = np.array(
                self.geometry
            ) * np.random.rand(2)
        else:
            self.bacteroid_locations[self.bacteroid_pointer, :] = np.array(location).astype(
                np.float64
            )

        if theta is None:
            theta = 2 * np.pi * np.random.rand() - np.pi
        else:
            theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        self.bacteroid_dirs[self.bacteroid_pointer] = theta

        self.bacteroid_age[self.bacteroid_pointer] = age
        self.bacteroid_energy[self.bacteroid_pointer] = energy
        self.bacteroid_is_seed[self.bacteroid_pointer] = is_seed
        self.bacteroid_is_stuck[self.bacteroid_pointer] = is_stuck
        self.bacteroid_remaining_attempts[self.bacteroid_pointer] = rem_attempt

        self.bacteroid_mask[self.bacteroid_pointer] = True
        self.num_bacteroids += 1
        self.bacteroid_pointer += 1

    def compact_bacteroid_arrays(self):
        self.bacteroid_locations[: self.num_bacteroids] = self.bacteroid_locations[
            self.bacteroid_mask
        ]
        self.bacteroid_dirs[: self.num_bacteroids] = self.bacteroid_dirs[self.bacteroid_mask]
        self.bacteroid_age[: self.num_bacteroids] = self.bacteroid_age[self.bacteroid_mask]
        self.bacteroid_energy[: self.num_bacteroids] = self.bacteroid_energy[self.bacteroid_mask]
        self.bacteroid_is_seed[: self.num_bacteroids] = self.bacteroid_is_seed[self.bacteroid_mask]
        self.bacteroid_is_stuck[: self.num_bacteroids] = self.bacteroid_is_stuck[
            self.bacteroid_mask
        ]
        self.bacteroid_remaining_attempts[: self.num_bacteroids] = (
            self.bacteroid_remaining_attempts[self.bacteroid_mask]
        )

        self.bacteroid_mask[: self.num_bacteroids] = True
        self.bacteroid_mask[self.num_bacteroids :] = False
        self.bacteroid_pointer = self.num_bacteroids

    def _expand_bacteroid_arrays(self):
        old_max_bacteroids = self.MAX_BACTEROIDS
        self.MAX_BACTEROIDS *= 2

        self.bacteroid_locations = np.pad(
            self.bacteroid_locations,
            pad_width=np.array(((0, old_max_bacteroids), (0, 0))),
            mode="constant",
            constant_values=(0, 0),
        )
        self.bacteroid_dirs = np.pad(
            self.bacteroid_dirs,
            pad_width=np.array((0, old_max_bacteroids)),
            mode="constant",
            constant_values=0.0,
        )
        self.bacteroid_age = np.pad(
            self.bacteroid_age,
            pad_width=np.array((0, old_max_bacteroids)),
            mode="constant",
            constant_values=0,
        )
        self.bacteroid_energy = np.pad(
            self.bacteroid_energy,
            pad_width=np.array((0, old_max_bacteroids)),
            mode="constant",
            constant_values=0.0,
        )
        self.bacteroid_is_seed = np.pad(
            self.bacteroid_is_seed,
            pad_width=np.array((0, old_max_bacteroids)),
            mode="constant",
            constant_values=False,
        )
        self.bacteroid_is_stuck = np.pad(
            self.bacteroid_is_stuck,
            pad_width=np.array((0, old_max_bacteroids)),
            mode="constant",
            constant_values=False,
        )
        self.bacteroid_remaining_attempts = np.pad(
            self.bacteroid_remaining_attempts,
            pad_width=np.array((0, old_max_bacteroids)),
            mode="constant",
            constant_values=0,
        )
        self.bacteroid_mask = np.pad(
            self.bacteroid_mask,
            pad_width=np.array((0, old_max_bacteroids)),
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
        # compute which class attributes should be saved
        rep = {attr: getattr(self, attr) for attr in dir(self) if attr[0] != "_"}
        rep = {k: v for k, v in rep.items() if isinstance(v, (int, float, bool, np.ndarray))}

        with h5py.File(filename, write_mode) as f:
            grp: h5py.Group = f.create_group(str(self.ticks))
            skip_list = set.union(
                *[
                    {f"num_{spec}s", f"{spec}_pointer", f"{spec}_mask"}
                    for spec in ["bifido", "bacteroid", "clost", "desulfo"]
                ]
            )
            for k, v in rep.items():
                # skip things that can be automatically reconstructed
                if k in skip_list:
                    continue

                if isinstance(v, (int, float, bool)):
                    # grp.create_dataset(k, shape=(), dtype=type(v), data=v, compression="gz")
                    grp.create_dataset(k, shape=(), dtype=type(v), data=v)
                else:
                    # numpy array
                    if np.issubdtype(v.dtype, np.object_):
                        v = v.astype(int)
                    for bact_type in ["bifido", "bacteroid", "clost", "desulfo"]:
                        if k.startswith(bact_type):
                            v = v[getattr(self, f"{bact_type}_mask")]
                            break
                    grp.create_dataset(k, shape=v.shape, dtype=v.dtype, data=v, compression="gz")

    @classmethod
    def load(cls, filename: str, time: int) -> "GutPython":
        """
        Instantiate the model from an HDF5 file.
        :param filename:
        :param time: which time slice to load from
        :return:
        """
        cell_types = ["bifido", "bacteroid", "clost", "desulfo"]
        non_cell_init_fields = [
            f
            for f in fields(cls)
            if f.init and not any(f.name.beginswith(cell_type) for cell_type in cell_types)
        ]
        cell_init_fields = [
            f
            for f in fields(cls)
            if f.init and any(f.name.beginswith(cell_type) for cell_type in cell_types)
        ]

        with h5py.File(filename, "r+") as f:
            grp: h5py.Group = f[str(time)]
            model = cls(
                **{f: grp[f][()] for f in non_cell_init_fields},
            )

            # scalars not initialized by init
            model.ticks = grp["time"][()]

            num_cells = {
                cell_type: grp[f"{cell_type}_locations"].shape[0] for cell_type in cell_types
            }
            # ensure there is enough space
            for cell_type in cell_types:
                if num_cells[cell_type] > getattr(model, f"{cell_type}_mask").size:
                    getattr(model, f"_expand_{cell_type}_arrays")()

            for field in cell_init_fields:
                cell_type = field.split("_")[0]
                model_field = getattr(model, field)
                model_field[: num_cells[cell_type]] = grp[field][()]

            for cell_type in cell_types:
                setattr(model, f"num_{cell_type}", num_cells[cell_type])
                setattr(model, f"{cell_type}_pointer", num_cells[cell_type])
                cell_mask = getattr(model, f"{cell_type}_mask")
                cell_mask[: num_cells[cell_type]] = True
                cell_mask[num_cells[cell_type] :] = False

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
                age=np.random.randint(1000),
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
                age=np.random.randint(1000),
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
                age=np.random.randint(1000),
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
                age=np.random.randint(1000),
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
        self.glucose_prev[:, :] = 0.0
        self.fo_prev[:, :] = 0.0
        self.lactose_prev[:, :] = 0.0
        self.lactate_prev[:, :] = 0.0
        self.inulin_prev[:, :] = 0.0
        self.cs_prev[:, :] = 0.0
        self.glucose_reserve[:, :] = 0.0
        self.fo_reserve[:, :] = 0.0
        self.lactose_reserve[:, :] = 0.0
        self.lactate_reserve[:, :] = 0.0
        self.inulin_reserve[:, :] = 0.0
        self.cs_reserve[:, :] = 0.0
        self.stuck_chance[:, :] = 0.0

        #   ;; setup the true absorption rate
        #   setTrueAbs

        self.set_true_abs()

        #   ;; setup the stuckChance
        #   setStuckChance

        self.set_stuck_chance()

        #   ;; Setup for stop if negative metas
        #   set negMeta false

        # self.neg_meta = False

        #   ;; set time to zero
        #   reset-ticks

        self.ticks = 0

        #   ;; reset the testState
        #   set testState 0

        self.test_state = 0

    def go(self):
        # to go
        # ;; This function determines the behavior at each time tick
        #
        #   ;; stop if error or unexpected output
        #   stopCheck

        self.stop_check()

        #   ;; Modify the energy level of each turtle and metabolite level of each patch
        #   ask patches [
        #     patchEat
        #     storeMetabolites
        #   ]

        self.patch_eat()
        self.store_metabolites()

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

        self.set_true_abs()

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

    def stop_check(self):
        # to stopCheck
        # ;; code for stopping the simulation on unexpected output
        #
        #   ;; Stop if negative number of metas calculated
        #   if negMeta [stop]
        #
        #   ;; Stop if any population hits 0 or there are too many turtles
        #   if (count turtles > 1000000) [ stop ]
        #   if not any? turtles [ stop ] ;; stop if all turtles are dead
        # end
        # if self.neg_meta:
        #     exit()
        if self.num_bacteria > 1000000:
            exit()
        if self.num_bacteria <= 0:
            exit()
        # TODO: implement better signalling than an immediate exit.

    def patch_eat(self):
        # to patchEat
        # ;; run this on a ask patches to have them start the turtle eating process
        #   ask turtles-here [
        #     set remAttempts 2 ;; reset the number of attempts
        #     set energy (energy - (100 / 1440)) ;; decrease the energy of the bacteria, currently survive 24 hours no
        #     ;; eat
        #   ]

        self.bacteroid_remaining_attempts[:] = 2
        self.bifido_remaining_attempts[:] = 2
        self.clost_remaining_attempts[:] = 2
        self.desulfo_remaining_attempts[:] = 2

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

        hungry_bacteroid_mask = (
            self.bacteroid_mask
            & (self.bacteroid_energy < 80)
            & (self.bacteroid_remaining_attempts > 0)
        )
        hungry_bifido_mask = (
            self.bifido_mask & (self.bifido_energy < 80) & (self.bifido_remaining_attempts > 0)
        )
        hungry_clost_mask = (
            self.clost_mask & (self.clost_energy < 80) & (self.clost_remaining_attempts > 0)
        )
        hungry_desulfo_mask = (
            self.desulfo_mask & (self.desulfo_energy < 80) & (self.desulfo_remaining_attempts > 0)
        )

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

            hungry_cell_idx = np.random.randint(len(hungry_bacteria))
            bact_type, idx = hungry_bacteria[hungry_cell_idx]

            metabolite_idx = np.random.randint(len(metabolites))
            metabolite = metabolites[metabolite_idx]

            self.bact_eat(bact_type, idx, metabolite)

            getattr(self, f"{bact_type}_remaining_attempts")[idx] -= 1
            if (
                getattr(self, f"{bact_type}_energy")[idx] >= 80
                or getattr(self, f"{bact_type}_remaining_attempts")[idx] <= 0
            ):
                hungry_bacteria.pop(hungry_cell_idx)

    def bact_eat(self, bact_type, idx, metabolite):
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
                loc = tuple(getattr(self, f"{bact_type}_locations")[idx].astype(int))
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
                loc = tuple(getattr(self, f"{bact_type}_locations")[idx].astype(int).T)
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
                loc = tuple(getattr(self, f"{bact_type}_locations")[idx].astype(int).T)
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
                loc = tuple(getattr(self, f"{bact_type}_locations")[idx].astype(int).T)
                if self.inulin[loc] >= 1:
                    getattr(self, f"{bact_type}_energy")[idx] += 25
                    self.inulin[loc] -= 1
                if bact_type == "bifido":
                    self.lactate[loc] += self.bifido_lactate_production
                pass
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
                loc = tuple(getattr(self, f"{bact_type}_locations")[idx].astype(int))
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
                loc = tuple(getattr(self, f"{bact_type}_locations")[idx].astype(int))
                if self.lactose[loc] >= 1:
                    getattr(self, f"{bact_type}_energy")[idx] += 25 if bact_type == "clost" else 50
                    self.lactose[loc] -= 1
                if bact_type == "bifido":
                    self.lactate[loc] += self.bifido_lactate_production
                pass
            case _:
                assert False

    def store_metabolites(self):
        # to storeMetabolites
        # ;; Sets previous metaohydrate variables to current levels to allow for correct
        # ;; transfer on ticks
        #   set inulinPrev ((inulin + inulinReserve))
        #   set FOPrev ((FO + FOReserve))
        #   set lactosePrev ((lactose + lactoseReserve))
        #   set lactatePrev ((lactate + lactateReserve))
        #   set glucosePrev ((glucose + glucoseReserve))
        #   set CSPrev ((CS + CSReserve))
        # end
        self.inulin_prev[:, :] = self.inulin + self.inulin_reserve
        self.fo_prev[:, :] = self.fo + self.fo_reserve
        self.lactose_prev[:, :] = self.lactose + self.lactose_reserve
        self.lactate_prev[:, :] = self.lactate + self.lactate_reserve
        self.glucose_prev[:, :] = self.glucose + self.glucose_reserve
        self.cs_prev[:, :] = self.cs + self.cs_reserve

    def make_metabolites(self):
        # to makeMetabolites
        # ;; Runs through all the metabolites and makes them, and moves them.
        #   let frac (flowDist - (floor( flowDist )))

        upper_flow_dist: int = math.ceil(self.flow_dist)
        lower_flow_dist: int = math.floor(self.flow_dist)
        frac: float = self.flow_dist - lower_flow_dist

        #   let span ((max-pycor - min-pycor) + 1)

        span: int = self.GRID_HEIGHT

        #   let leftDist (pxcor - min-pxcor)
        #
        #   if ((inulin < 0) or (CS < 0) or (FO < 0) or (lactose < 0) or (lactate < 0) or (glucose < 0)) [
        #     print "ERROR! Patch reported negative metabolite. Problem with simulation leading to inaccurate results.
        #     Terminating Program."
        #     set negMeta true
        #     stop
        #   ]
        #
        #   set inulin ((inulin) + inulinReserve)
        #   set FO ((FO) + FOReserve)
        #   set lactose ((lactose) + lactoseReserve)
        #   set lactate ((lactate) + lactateReserve)
        #   set glucose ((glucose) + glucoseReserve)
        #   set CS ((CS) + CSReserve)
        #
        #   let remainFactor 0
        #   if (flowDist < 1)[set remainFactor (1 - flowDist)]
        #   set inulin (inulin * remainFactor)
        #   set FO (FO * remainFactor)
        #   set lactose (lactose * remainFactor)
        #   set lactate (lactate * remainFactor)
        #   set glucose (glucose * remainFactor)
        #   set CS (CS * remainFactor)
        #
        #   ;;The leftmost patches evenly split the inFlow number of metas
        #   ifelse (leftDist < flowDist)[
        #     let inFlowCoef (((min list 1 (flowDist - leftDist))) / (flowDist * span))
        #     set inulin ((inulin) + (inFlowInulin * inFlowCoef))
        #     set FO ((FO) + (inFlowFO * inFlowCoef))
        #     set lactose ((lactose) + (inFlowLactose * inFlowCoef))
        #     set lactate ((lactate) + (inFlowLactate * inFlowCoef))
        #     set glucose ((glucose) + (inFlowGlucose * inFlowCoef))
        #     set CS ((CS) + (inFlowCS * inFlowCoef))
        #   ]
        #   [
        #     let added ( ((get-inulin (- (ceiling flowDist)) 0) * (min list frac (1 - remainFactor)))
        #     + ((get-inulin (- (floor flowDist)) 0) * (min list (1 - frac) (floor flowDist))) )
        #     ifelse (inulin + added) < 1000[
        #       set inulin (inulin + (added))
        #     ]
        # 		[
        # 			set inulin (1000)
        # 		]

        in_flow_coef = np.clip(
            (self.flow_dist - np.arange(upper_flow_dist)) / (self.flow_dist * span), 0, 1
        )[:, np.newaxis]
        remain_factor = 0 if self.flow_dist >= 1 else 1 - self.flow_dist

        for metabolite, metabolite_prev, metabolite_reserve, metabolite_inflow in [
            (self.inulin, self.inulin_prev, self.inulin_reserve, self.inulin_inflow),
            (self.fo, self.fo_prev, self.fo_reserve, self.fo_inflow),
            (self.lactose, self.lactose_prev, self.lactose_reserve, self.lactose_inflow),
            (self.lactate, self.lactate_prev, self.lactate_reserve, self.lactate_inflow),
            (self.glucose, self.glucose_prev, self.glucose_reserve, self.glucose_inflow),
            (self.cs, self.cs_prev, self.cs_reserve, self.cs_inflow),
        ]:
            metabolite += metabolite_reserve
            metabolite *= remain_factor

            metabolite[:upper_flow_dist, :] += in_flow_coef * metabolite_inflow

            if frac == 0.0:
                metabolite[upper_flow_dist:, :] += metabolite_prev[:-upper_flow_dist, :] * (
                    1 - remain_factor
                )
            elif lower_flow_dist == 0:
                metabolite[upper_flow_dist:, :] = metabolite_prev[:-upper_flow_dist, :] * frac * (
                    1 - remain_factor
                ) + metabolite_prev[1:, :] * (1 - frac) * (1 - remain_factor)
            else:
                metabolite[upper_flow_dist:, :] = metabolite_prev[:-upper_flow_dist, :] * frac * (
                    1 - remain_factor
                ) + metabolite_prev[1:-lower_flow_dist, :] * (1 - frac) * (1 - remain_factor)
            np.clip(metabolite, 0, 1000, out=metabolite)
            metabolite[metabolite < 0.001] = 0

        # ;;Need to handle case of patch which flowDist ends in from beginning
        # ACK: I think that I already have? TODO: check

        # 	ifelse (((max-pxcor - min-pxcor) < 1))[
        # 		set inulinReserve (0)
        #   	set FOReserve (0)
        #   	set lactoseReserve (0)
        #   	set lactateReserve (0)
        #   	set glucoseReserve (0)
        #   	set CSReserve (0)
        # 	][
        #   	set inulinReserve ((inulin) * reserveFraction * ((max-pxcor - pxcor)/(max-pxcor - min-pxcor)))
        #   	set FOReserve ((FO) * reserveFraction * ((max-pxcor - pxcor)/(max-pxcor - min-pxcor)))
        #   	set lactoseReserve ((lactose) * reserveFraction * ((max-pxcor - pxcor)/(max-pxcor - min-pxcor)))
        #   	set lactateReserve ((lactate) * reserveFraction * ((max-pxcor - pxcor)/(max-pxcor - min-pxcor)))
        #   	set glucoseReserve ((glucose) * reserveFraction * ((max-pxcor - pxcor)/(max-pxcor - min-pxcor)))
        #   	set CSReserve ((CS) * reserveFraction * ((max-pxcor - pxcor)/(max-pxcor - min-pxcor)))
        # 	]
        #
        #   	set inulin ((inulin - inulinReserve) * (1 - trueAbsorption))
        #   	set FO ((FO - FOReserve) * (1 - trueAbsorption))
        #   	set lactose ((lactose - lactoseReserve) * (1 - trueAbsorption))
        #   	set lactate ((lactate - lactateReserve) * (1 - trueAbsorption))
        #   	set glucose ((glucose - glucoseReserve) * (1 - trueAbsorption))
        #   	set CS ((CS - CSReserve) * (1 - trueAbsorption))

        for metabolite, metabolite_reserve in [
            (self.inulin, self.inulin_reserve),
            (self.fo, self.fo_reserve),
            (self.lactose, self.lactose_reserve),
            (self.lactate, self.lactate_reserve),
            (self.glucose, self.glucose_reserve),
            (self.cs, self.cs_reserve),
        ]:
            if self.GRID_WIDTH == 1:
                metabolite_reserve[:, :] = 0.0
            else:
                metabolite_reserve[:, :] = (
                    metabolite
                    * self.reserve_fraction
                    * np.arange(self.GRID_WIDTH)[::-1, np.newaxis]
                    / (self.GRID_WIDTH - 1)
                )
            metabolite[:, :] = (metabolite - metabolite_reserve) * (1 - self.true_absorption)

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
        bifido_excrete = (self.bifido_locations[:, 0] >= self.GRID_WIDTH) & self.bifido_mask
        self.bifido_mask[bifido_excrete] = False
        self.num_bifidos -= np.sum(bifido_excrete)

        # deathBifidos
        to_kill = self.bifido_mask & (self.bifido_energy <= 0)
        self.bifido_mask[to_kill] = False
        self.num_bifidos -= np.sum(to_kill)

        # checkStuck
        bifido_locs = tuple(
            np.minimum(self.bifido_locations.astype(np.int64), np.array(self.geometry) - 1).T
        )
        sticking_bifidos = (
            self.bifido_mask
            & ~self.bifido_is_stuck
            & (np.random.rand(*self.bifido_mask.shape) < (self.stuck_chance[bifido_locs] / 100.0))
        )
        self.bifido_is_stuck[sticking_bifidos] = True
        unsticking_bifidos = (
            self.bifido_mask
            & self.bifido_is_stuck
            & (np.random.rand(self.bifido_is_stuck.shape[0]) < (self.unstuck_chance / 100.0))
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
        desulfo_excrete = (self.desulfo_locations[:, 0] >= self.GRID_WIDTH) & self.desulfo_mask
        self.desulfo_mask[desulfo_excrete] = False
        self.num_desulfos -= np.sum(desulfo_excrete)

        # deathDesulfos
        to_kill = self.desulfo_mask & (self.desulfo_energy <= 0)
        self.desulfo_mask[to_kill] = False
        self.num_desulfos -= np.sum(to_kill)

        # checkStuck
        desulfo_locs = tuple(
            np.minimum(self.desulfo_locations.astype(np.int64), np.array(self.geometry) - 1).T
        )
        sticking_desulfos = (
            self.desulfo_mask
            & ~self.desulfo_is_stuck
            & (
                np.random.rand(*self.desulfo_is_stuck.shape)
                < (self.stuck_chance[desulfo_locs] / 100.0)
            )
        )
        self.desulfo_is_stuck[sticking_desulfos] = True
        unsticking_desulfos = (
            self.desulfo_mask
            & self.desulfo_is_stuck
            & (np.random.rand(*self.desulfo_is_stuck.shape) < (self.unstuck_chance / 100.0))
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
        clost_excrete = (self.clost_locations[:, 0] >= self.GRID_WIDTH) & self.clost_mask
        self.clost_mask[clost_excrete] = False
        self.num_closts -= np.sum(clost_excrete)

        # deathClosts
        to_kill = self.clost_mask & (self.clost_energy <= 0)
        self.clost_mask[to_kill] = False
        self.num_closts -= np.sum(to_kill)

        # checkStuck
        clost_locs = tuple(
            np.minimum(self.clost_locations.astype(np.int64), np.array(self.geometry) - 1).T
        )
        sticking_closts = (
            self.clost_mask
            & ~self.clost_is_stuck
            & (np.random.rand(*self.clost_mask.shape) < (self.stuck_chance[clost_locs] / 100.0))
        )
        self.clost_is_stuck[sticking_closts] = True
        unsticking_closts = (
            self.clost_mask
            & self.clost_is_stuck
            & (np.random.rand(*self.clost_mask.shape) < (self.unstuck_chance / 100.0))
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
            self.bacteroid_locations[:, 0] >= self.GRID_WIDTH
        ) & self.bacteroid_mask
        self.bacteroid_mask[bacteroid_excrete] = False
        self.num_bacteroids -= np.sum(bacteroid_excrete)

        # deathBacteroids
        to_kill = self.bacteroid_mask & (self.bacteroid_energy <= 0)
        self.bacteroid_mask[to_kill] = False
        self.num_bacteroids -= np.sum(to_kill)

        # checkStuck
        bacteroid_locs = tuple(
            np.minimum(self.bacteroid_locations.astype(np.int64), np.array(self.geometry) - 1).T
        )
        sticking_bacteroids = (
            self.bacteroid_mask
            & ~self.bacteroid_is_stuck
            & (
                np.random.rand(self.bacteroid_is_stuck.shape[0])
                < (self.stuck_chance[bacteroid_locs] / 100.0)
            )
        )
        self.bacteroid_is_stuck[sticking_bacteroids] = True
        unsticking_bacteroids = (
            self.bacteroid_mask
            & self.bacteroid_is_stuck
            & (np.random.rand(self.bacteroid_is_stuck.shape[0]) < (self.unstuck_chance / 100.0))
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
            & (np.random.rand(self.bacteroid_mask.shape[0]) < (self.seed_chance / 100.0))
        ] = True

        self.bifido_is_seed[
            self.bifido_is_stuck
            & (np.random.rand(self.bifido_mask.shape[0]) < (self.seed_chance / 100.0))
        ] = True

        self.clost_is_seed[
            self.clost_is_stuck
            & (np.random.rand(self.clost_mask.shape[0]) < (self.seed_chance / 100.0))
        ] = True

        self.desulfo_is_seed[
            self.desulfo_is_stuck
            & (np.random.rand(self.desulfo_mask.shape[0]) < (self.seed_chance / 100.0))
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

        for _ in range(self.in_conc_bifidos):
            self.create_bifido(
                energy=100,
                is_seed=False,
                is_stuck=False,
                age=np.random.randint(1000),
                location=[0.0, np.random.rand()],
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

        for _ in range(self.in_conc_desulfos):
            self.create_desulfo(
                energy=100,
                is_seed=False,
                # is_stuck=False,
                age=np.random.randint(1000),
                location=[0.0, np.random.rand()],
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

        for _ in range(self.in_conc_closts):
            self.create_clost(
                energy=100,
                is_seed=False,
                is_stuck=False,
                age=np.random.randint(1000),
                location=[0.0, np.random.rand()],
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

        for _ in range(self.in_conc_bacteroids):
            self.create_bacteroid(
                energy=100,
                is_seed=False,
                is_stuck=False,
                age=np.random.randint(1000),
                location=[0.0, np.random.rand()],
            )

    def set_true_abs(self):
        # to setTrueAbs
        #   ;; controls the true absorption rate
        #
        #   ;; 0.723823204 is the weighted average immune response coefficient calculated for
        #   ;; Healthy bacteria gut percentages. This allows the absorption to change due to
        #   ;; bacteria populations, simulating immune response.
        #
        # 	ifelse (any? turtles)[
        #   	set trueAbsorption absorption * (0.723823204 / ((0.8 * ((count desulfos) / (count turtles))) +
        #   	(1 * ((count closts) / (count turtles)))+(1.2 * ((count bacteroides) / (count turtles))) +
        #   	(0.7 * ((count bifidos) / (count turtles)))))
        # 	][
        # 		set trueAbsorption 0
        # print "ERROR! Bacteria died out. Problem with simulation leading to inaccurate results. Terminating Program."
        # 	]
        # end

        total_bacteria = (
            self.num_bacteroids + self.num_bifidos + self.num_closts + self.num_desulfos
        )
        assert total_bacteria > 0, "Bacteria died out!"

        # TODO: package the constants
        self.true_absorption = self.absorption * (
            self.absorption_constant
            / (
                (0.8 * (self.num_desulfos / total_bacteria))
                + (1 * (self.num_closts / total_bacteria))
                + (1.2 * (self.num_bacteroids / total_bacteria))
                + (0.7 * (self.num_bifidos / total_bacteria))
            )
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


if __name__ == "__main__":

    gp = GutPython()
    gp.setup()
    for _ in range(10_000):
        gp.go()
