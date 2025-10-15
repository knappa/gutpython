from typing import Iterable, Optional, Tuple

import numpy as np
from attr import define, field

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
    # static properties

    @property
    def geometry(self) -> Tuple[int, int]:
        return self.GRID_HEIGHT, self.GRID_WIDTH

    ######################################################################
    # bifidobacteria

    bifido_doub_const: float = field(default=1.0)
    bifido_flow_const: float = field(default=1.0)

    num_bifidos: int = field(init=False, factory=lambda: 0, type=int)
    bifido_pointer: int = field(init=False, factory=lambda: 0, type=int)

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

    bifido_rem_attempts = field(type=np.ndarray)

    @bifido_rem_attempts.default
    def _bifido_rem_attempts_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=np.int64)

    bifido_excrete = field(type=np.ndarray)

    @bifido_excrete.default
    def _bifido_excrete_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=np.bool_)

    bifido_energy = field(type=np.ndarray)

    @bifido_energy.default
    def _bifido_energy_factory(self):
        return np.zeros(self.MAX_BIFIDOS, dtype=np.float64)

    ######################################################################
    # desulfovibro

    desulfo_doub_const: float = field(default=1.0)
    desulfo_flow_const: float = field(default=1.0)

    num_desulfos: int = field(init=False, factory=lambda: 0, type=int)
    desulfo_pointer: int = field(init=False, factory=lambda: 0, type=int)

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

    desulfo_rem_attempts = field(type=np.ndarray)

    @desulfo_rem_attempts.default
    def _desulfo_rem_attempts_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=np.int64)

    desulfo_excrete = field(type=np.ndarray)

    @desulfo_excrete.default
    def _desulfo_excrete_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=np.bool_)

    desulfo_energy = field(type=np.ndarray)

    @desulfo_energy.default
    def _desulfo_energy_factory(self):
        return np.zeros(self.MAX_DESULFOS, dtype=np.float64)

    ######################################################################
    # clostridia

    clost_doub_const: float = field(default=1.0)
    clost_flow_const: float = field(default=1.0)

    num_closts: int = field(init=False, factory=lambda: 0, type=int)
    clost_pointer: int = field(init=False, factory=lambda: 0, type=int)

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

    clost_rem_attempts = field(type=np.ndarray)

    @clost_rem_attempts.default
    def _clost_rem_attempts_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=np.int64)

    clost_excrete = field(type=np.ndarray)

    @clost_excrete.default
    def _clost_excrete_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=np.bool_)

    clost_energy = field(type=np.ndarray)

    @clost_energy.default
    def _clost_energy_factory(self):
        return np.zeros(self.MAX_CLOSTS, dtype=np.float64)

    ######################################################################
    # bacteriodes

    bacteroid_doub_const: float = field(default=1.0)
    bacteroid_flow_const: float = field(default=1.0)

    num_bacteroids: int = field(init=False, factory=lambda: 0, type=int)
    bacteroid_pointer: int = field(init=False, factory=lambda: 0, type=int)

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

    bacteroid_rem_attempts = field(type=np.ndarray)

    @bacteroid_rem_attempts.default
    def _bacteroid_rem_attempts_factory(self):
        return np.zeros(self.MAX_BACTEROIDS, dtype=np.int64)

    bacteroid_excrete = field(type=np.ndarray)

    @bacteroid_excrete.default
    def _bacteroid_excrete_factory(self):
        return np.zeros(self.MAX_BACTEROIDS, dtype=np.bool_)

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

    ava_metas = field(type=np.ndarray)

    @ava_metas.default
    def _ava_metas_factory(self):
        return np.full(self.geometry, 0.0, dtype=np.float64)

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
        excrete: Optional[bool] = None,
        is_seed: Optional[bool] = None,
        is_stuck: Optional[bool] = None,
        rem_attempt: Optional[int] = None,
    ) -> None:
        """
        Create a Bifidobacterium.

        :param location: location to create the bifidobacterium (optional, random if omitted)
        :param theta: direction of bifidobacterium movement in radians (optional, random if omitted)
        :param age:
        :param rem_attempt:
        :param is_stuck:
        :param is_seed:
        :param excrete:
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
        self.bifido_excrete[self.bifido_pointer] = excrete
        self.bifido_is_seed[self.bifido_pointer] = is_seed
        self.bifido_is_stuck[self.bifido_pointer] = is_stuck
        self.bifido_rem_attempts[self.bifido_pointer] = rem_attempt

        self.bifido_mask[self.bifido_pointer] = True
        self.num_bifidos += 1
        self.bifido_pointer += 1

    def compact_bifido_arrays(self):
        self.bifido_locations[: self.num_bifidos] = self.bifido_locations[self.bifido_mask]
        self.bifido_dirs[: self.num_bifidos] = self.bifido_dirs[self.bifido_mask]
        self.bifido_age[: self.num_bifidos] = self.bifido_age[self.bifido_mask]
        self.bifido_energy[: self.num_bifidos] = self.bifido_energy[self.bifido_mask]
        self.bifido_excrete[: self.num_bifidos] = self.bifido_excrete[self.bifido_mask]
        self.bifido_is_seed[: self.num_bifidos] = self.bifido_is_seed[self.bifido_mask]
        self.bifido_is_stuck[: self.num_bifidos] = self.bifido_is_stuck[self.bifido_mask]
        self.bifido_rem_attempts[: self.num_bifidos] = self.bifido_rem_attempts[self.bifido_mask]

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
        self.bifido_excrete = np.pad(
            self.bifido_excrete,
            pad_width=np.array((0, old_max_bifidos)),
            mode="constant",
            constant_values=False,
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
        self.bifido_rem_attempts = np.pad(
            self.bifido_rem_attempts,
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
