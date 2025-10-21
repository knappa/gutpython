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
    # parameters from interface

    max_stuck_chance: float = field(default=50)  # TODO: understand units. percent?
    low_stuck_bound: float = field(default=2)  # TODO: understand units. percent?
    unstuck_chance: float = field(default=10)  # TODO: understand units. percent?
    mid_stuck_conc: float = field(default=10.0)  # TODO: understand units. percent?
    seed_chance: float = field(default=5.0)  # TODO: understand units. percent?
    seed_percent: float = field(default=5.0)

    init_num_bifidos: int = field(default=23562)  # TODO: uhh? Seems awfully specific.
    init_num_bacteroids: int = field(default=5490)  # TODO: uhh? Seems awfully specific.
    init_num_closts: int = field(default=921)  # TODO: uhh? Seems awfully specific.
    init_num_desulfos: int = field(default=70)

    ######################################################################
    # other globals

    neg_meta: bool = False

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
    # bacteroides

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

    ######################################################################
    # Desulfovibro utility functions

    def create_desulfo(
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
        Create a Desulfovibro.

        :param location: location to create the desulfovibro (optional, random if omitted)
        :param theta: direction of desulfovibro movement in radians (optional, random if omitted)
        :param age:
        :param rem_attempt:
        :param is_stuck:
        :param is_seed:
        :param excrete:
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
        self.desulfo_excrete[self.desulfo_pointer] = excrete
        self.desulfo_is_seed[self.desulfo_pointer] = is_seed
        self.desulfo_is_stuck[self.desulfo_pointer] = is_stuck
        self.desulfo_rem_attempts[self.desulfo_pointer] = rem_attempt

        self.desulfo_mask[self.desulfo_pointer] = True
        self.num_desulfos += 1
        self.desulfo_pointer += 1

    def compact_desulfo_arrays(self):
        self.desulfo_locations[: self.num_desulfos] = self.desulfo_locations[self.desulfo_mask]
        self.desulfo_dirs[: self.num_desulfos] = self.desulfo_dirs[self.desulfo_mask]
        self.desulfo_age[: self.num_desulfos] = self.desulfo_age[self.desulfo_mask]
        self.desulfo_energy[: self.num_desulfos] = self.desulfo_energy[self.desulfo_mask]
        self.desulfo_excrete[: self.num_desulfos] = self.desulfo_excrete[self.desulfo_mask]
        self.desulfo_is_seed[: self.num_desulfos] = self.desulfo_is_seed[self.desulfo_mask]
        self.desulfo_is_stuck[: self.num_desulfos] = self.desulfo_is_stuck[self.desulfo_mask]
        self.desulfo_rem_attempts[: self.num_desulfos] = self.desulfo_rem_attempts[
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
        self.desulfo_excrete = np.pad(
            self.desulfo_excrete,
            pad_width=np.array((0, old_max_desulfos)),
            mode="constant",
            constant_values=False,
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
        self.desulfo_rem_attempts = np.pad(
            self.desulfo_rem_attempts,
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
        excrete: Optional[bool] = None,
        is_seed: Optional[bool] = None,
        is_stuck: Optional[bool] = None,
        rem_attempt: Optional[int] = None,
    ) -> None:
        """
        Create a Clostridium.

        :param location: location to create the clostridium (optional, random if omitted)
        :param theta: direction of clostridium movement in radians (optional, random if omitted)
        :param age:
        :param rem_attempt:
        :param is_stuck:
        :param is_seed:
        :param excrete:
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
        self.clost_excrete[self.clost_pointer] = excrete
        self.clost_is_seed[self.clost_pointer] = is_seed
        self.clost_is_stuck[self.clost_pointer] = is_stuck
        self.clost_rem_attempts[self.clost_pointer] = rem_attempt

        self.clost_mask[self.clost_pointer] = True
        self.num_closts += 1
        self.clost_pointer += 1

    def compact_clost_arrays(self):
        self.clost_locations[: self.num_closts] = self.clost_locations[self.clost_mask]
        self.clost_dirs[: self.num_closts] = self.clost_dirs[self.clost_mask]
        self.clost_age[: self.num_closts] = self.clost_age[self.clost_mask]
        self.clost_energy[: self.num_closts] = self.clost_energy[self.clost_mask]
        self.clost_excrete[: self.num_closts] = self.clost_excrete[self.clost_mask]
        self.clost_is_seed[: self.num_closts] = self.clost_is_seed[self.clost_mask]
        self.clost_is_stuck[: self.num_closts] = self.clost_is_stuck[self.clost_mask]
        self.clost_rem_attempts[: self.num_closts] = self.clost_rem_attempts[self.clost_mask]

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
        self.clost_excrete = np.pad(
            self.clost_excrete,
            pad_width=np.array((0, old_max_closts)),
            mode="constant",
            constant_values=False,
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
        self.clost_rem_attempts = np.pad(
            self.clost_rem_attempts,
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
        excrete: Optional[bool] = None,
        is_seed: Optional[bool] = None,
        is_stuck: Optional[bool] = None,
        rem_attempt: Optional[int] = None,
    ) -> None:
        """
        Create a Bacteroides.

        :param location: location to create the bacteroides (optional, random if omitted)
        :param theta: direction of bacteroides movement in radians (optional, random if omitted)
        :param age:
        :param rem_attempt:
        :param is_stuck:
        :param is_seed:
        :param excrete:
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
        self.bacteroid_excrete[self.bacteroid_pointer] = excrete
        self.bacteroid_is_seed[self.bacteroid_pointer] = is_seed
        self.bacteroid_is_stuck[self.bacteroid_pointer] = is_stuck
        self.bacteroid_rem_attempts[self.bacteroid_pointer] = rem_attempt

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
        self.bacteroid_excrete[: self.num_bacteroids] = self.bacteroid_excrete[self.bacteroid_mask]
        self.bacteroid_is_seed[: self.num_bacteroids] = self.bacteroid_is_seed[self.bacteroid_mask]
        self.bacteroid_is_stuck[: self.num_bacteroids] = self.bacteroid_is_stuck[
            self.bacteroid_mask
        ]
        self.bacteroid_rem_attempts[: self.num_bacteroids] = self.bacteroid_rem_attempts[
            self.bacteroid_mask
        ]

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
        self.bacteroid_excrete = np.pad(
            self.bacteroid_excrete,
            pad_width=np.array((0, old_max_bacteroids)),
            mode="constant",
            constant_values=False,
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
        self.bacteroid_rem_attempts = np.pad(
            self.bacteroid_rem_attempts,
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
    # initialization code

    def __attrs_post_init__(self):
        self.setup()

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
                excrete=False,
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
                excrete=False,
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
                excrete=False,
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
                excrete=False,
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

        self.neg_meta = False

        #   ;; set time to zero
        #   reset-ticks

        # TODO

        #   ;; reset the testState
        #   set testState 0

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
        pass

    def set_stuck_chance(self):
        occupancy = np.zeros(self.geometry, dtype=np.int64)

        bifido_patches = self.bifido_locations[self.bifido_mask, :].astype(np.int64)
        # TODO: vectorize. need to check what happens with repeat locs using
        #  occupancy[tuple(bifido_patches.T)] += 1
        for idx in range(bifido_patches.shape[0]):
            occupancy[tuple(bifido_patches[idx])] += 1

        desulfo_patches = self.desulfo_locations[self.desulfo_mask, :].astype(np.int64)
        for idx in range(desulfo_patches.shape[0]):
            occupancy[tuple(desulfo_patches[idx])] += 1

        clost_patches = self.clost_locations[self.clost_mask, :].astype(np.int64)
        for idx in range(clost_patches.shape[0]):
            occupancy[tuple(clost_patches[idx])] += 1

        bacteroid_patches = self.bacteroid_locations[self.bacteroid_mask, :].astype(np.int64)
        for idx in range(bacteroid_patches.shape[0]):
            occupancy[tuple(bacteroid_patches[idx])] += 1

        self.stuck_chance[:, :] = self.max_stuck_chance * (
            1 - occupancy / (self.mid_stuck_conc + occupancy)
        )
        self.stuck_chance[self.stuck_chance < self.low_stuck_bound] = 0
