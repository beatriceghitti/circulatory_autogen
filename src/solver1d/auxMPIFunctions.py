#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI communication between the 1d solver (main1D.py) and the 1d-0d coupler (src/coupler/coupler.cpp).

The coupler, the 0d solver and this 1d solver are started together by a single mpirun
(MPMD launch, see src/coupler/launch_coupled_mpi.bash). All messages go through the coupler.

Uses the uppercase, buffer-based mpi4py methods (Send/Recv/...) with explicit dtypes, because
they send raw numbers that C++ can read. The lowercase methods (send/recv/...) pickle Python
objects, which the C++ side cannot decode.

The constants below must match src/coupler/coupler_mpi_common.h. KEEP THEM IN SYNC.
"""

import math
import sys
import traceback
import numpy as np

# Must match coupler_mpi_common.h
COLOR_COUPLER, COLOR_SOLVER0D, COLOR_SOLVER1D = 0, 1, 2
TAG_CONFIG, TAG_DONE, TAG_CONN, TAG_DT, TAG_DT_STEP, TAG_DATA, TAG_VOL = 1, 2, 3, 10, 11, 20, 30
DATA_LENGTH = 2     # doubles per time/time step message and per 1d-0d connection
CONFIG_LENGTH = 10  # doubles in the TAG_CONFIG handshake:
                    # {N1d0d, number of global connections, T0, pre_time, sim_time, tolTime, tSave, tEnd,
                    #  PETSc scheme code, N1d0dTot}
# Fields of a 1d-0d connection sent by the coupler right after the handshake (TAG_CONN), in this order, -1 for a field
# that the connection does not have (COUPLING_CONN_FIELDS in coupler_mpi_common.h)
CONN_FIELDS = ("vess1d_idx", "vess1d_bc_in0_or_out1", "vess0d_idx", "cellml_idx", "cellml_bc_in0_or_out1",
               "cellml_bc_flow0_or_press1", "port_idx", "port_flow0_or_press1", "port_state0_or_var1", "R_T_variable_idx",
               "pressure_bc_as_flow_bc", "port_volume_sum")
# the fields used by the 1d solver, compared with its own copy of the connections file (the others are used only by the
# 0d solver, so that a 1d copy from another generation is accepted if the 1d side of the connections is the same)
CONN_FIELDS_1D = ("vess1d_idx", "vess1d_bc_in0_or_out1", "cellml_bc_in0_or_out1", "cellml_bc_flow0_or_press1",
                  "port_flow0_or_press1", "port_volume_sum")
# Types of 1d-0d connections (CouplingConnectionType in coupler_mpi_common.h, see "Global connections" there):
# - one-to-one: between a 1d vessel end and a 0d module, exchanged in TAG_DATA;
# - global (vess1d_idx = -1), after all the one-to-one connections, exchanged in TAG_VOL (one message, one row of
#   DATA_LENGTH values per global connection, at start-up and at the end of each global time step). The only type of
#   global connection so far is the volume sum (port_volume_sum = 1): {total blood volume of the 1d model [m3], 0}.
CONN_ONE_TO_ONE, CONN_GLOBAL_VOLUME_SUM = 0, 1


def connection_type(conn):
    """
    Type of a connection of the connections file (CONN_ONE_TO_ONE or CONN_GLOBAL_VOLUME_SUM), as
    coupling_connection_type() in coupler_mpi_common.h. KEEP THEM IN SYNC. Raises ValueError with the reason if the
    connection has none of these types.
    """
    if "vess1d_idx" not in conn:
        raise ValueError("the connection has no vess1d_idx field")
    volume_sum = conn.get("port_volume_sum", -1) == 1
    if conn["vess1d_idx"] >= 0:
        if volume_sum:
            raise ValueError("a connection with a 1d vessel (vess1d_idx >= 0) cannot be a volume sum connection")
        return CONN_ONE_TO_ONE
    if volume_sum:
        return CONN_GLOBAL_VOLUME_SUM
    raise ValueError("unknown type of global connection (vess1d_idx = -1): the only type of global connection is the "
                     "volume sum (port_volume_sum = 1)")


# Codes of the PETSc time integration schemes sent in the handshake (CouplingPetscScheme in coupler_mpi_common.h):
# 0 when the ODE solver is not PETSc
PETSC_SCHEME_NONE = 0
PETSC_SCHEME_CODES = {"BDF1": 1, "CN": 2, "BDF2": 3}

# The coupling currently in use (if any), so that errors can abort the whole MPI job.
_active_coupling = None


# Rounding of times to the grid of the time tolerance tolTime: must give exactly the same result as
# coupling_round_time() in coupler_mpi_common.h, used by the coupler and the 0d solver. KEEP THEM IN SYNC.
# Used also in standalone 1d runs (it does not need MPI).

def _round_half_away(x):
    """Round to the nearest integer, halves away from zero, as std::round in C++ (Python's round() rounds halves to even)."""
    a = abs(float(x))
    q = math.floor(a)           # a - q is exact in floating point
    if a - q >= 0.5:
        q += 1
    return math.copysign(float(q), x)


def inv_tol_time(tolTime):
    """Inverse of the time tolerance, rounded to an integer (1e11 for 1e-11), as COUPLING_INV_TOL_TIME."""
    return _round_half_away(1.0 / tolTime)


def round_time(t, invTolTime):
    """Round time t to the grid of tolTime = 1/invTolTime, as coupling_round_time() in coupler_mpi_common.h."""
    return _round_half_away(t * invTolTime) / invTolTime


class MPICoupling1D:
    """Coupled mode of the 1d solver: exchanges messages with the coupler over MPI."""

    def __init__(self):
        global _active_coupling
        import mpi4py
        mpi4py.rc.thread_level = "single"   # match the C++ MPI_Init
        from mpi4py import MPI  # importing initialises MPI
        self.MPI = MPI
        self.world = MPI.COMM_WORLD

        # Same two collective calls, in the same order, as setup_coupling() in coupler_mpi_common.h.
        # This must happen before anything that can fail, so that a failure aborts the whole job.
        self.local = self.world.Split(COLOR_SOLVER1D, 0)
        colors = np.empty(self.world.Get_size(), dtype=np.intc)
        self.world.Allgather(np.array([COLOR_SOLVER1D], dtype=np.intc), colors)
        _active_coupling = self

        if COLOR_COUPLER not in colors:
            self.abort("1d solver :: Error: coupled mode requested, but no coupler in this mpirun")
        self.coupler = int(np.flatnonzero(colors == COLOR_COUPLER)[0])
        if self.local.Get_size() != 1:
            self.abort(f"1d solver :: Error: the 1d solver must run on exactly 1 MPI rank, not {self.local.Get_size()}")

        self.N1d0d = 0
        self.N1d0dGlobal = 0
        print(f"1d solver :: MPI coupling initialised, coupler is world rank {self.coupler}", flush=True)

    def handshake(self, N1d0d, N1d0dGlobal, T0, pre_time, sim_time, petsc_solver, N1d0dTot):
        """
        Receive the coupling layout from the coupler and check that it matches ours.

        N1d0dGlobal: number of global connections (e.g. volume sum), the rows of the TAG_VOL messages.
        petsc_solver: PETSc time integration scheme of the 1d solver ("BDF1", "CN" or "BDF2", PETSc_solver in input.ini),
        or None if its ODE solver is not PETSc: it must be the scheme of the coupler configuration file, with which
        the 0d solver was built (see CouplingPetscScheme in coupler_mpi_common.h).
        N1d0dTot: number of connections (one-to-one and global) in the 1d copy of the connections file.

        Returns the time tolerance, the save time and the end time of the coupled simulation (COUPLING_TOL_TIME,
        coupling_save_time() and coupling_end_time() in coupler_mpi_common.h), which the 1d solver must use
        in a coupled run.
        """
        # First check the length: with a coupler built from another version of coupler_mpi_common.h,
        # the missing values of cfg would be left uninitialised (or MPI would report a truncation error).
        status = self.MPI.Status()
        self.world.Probe(source=self.coupler, tag=TAG_CONFIG, status=status)
        n = status.Get_count(self.MPI.DOUBLE)
        if n != CONFIG_LENGTH:
            self.abort(f"1d solver :: Error: the coupler sent {n} values in the handshake instead of {CONFIG_LENGTH}: "
                       "the coupler and this 1d solver come from different versions (rebuild the coupler)")
        cfg = np.empty(CONFIG_LENGTH, dtype=np.float64)
        self.world.Recv(cfg, source=self.coupler, tag=TAG_CONFIG)
        mine = np.array([N1d0d, N1d0dGlobal, T0, pre_time, sim_time], dtype=np.float64)
        if not np.allclose(cfg[:5], mine, rtol=1e-12, atol=0.0):
            self.abort("1d solver :: Error: coupling mismatch with the coupler "
                       f"[N1d0d, number of global connections, T0, pre_time, sim_time]: coupler {cfg[:5].tolist()} "
                       f"vs 1d solver {mine.tolist()}")
        tolTime = float(cfg[5])
        if not tolTime > 0.0:
            self.abort(f"1d solver :: Error: invalid time tolerance from the coupler: {tolTime}")
        tSave, tEnd = float(cfg[6]), float(cfg[7])
        # the coupler rounds pre_time and pre_time + sim_time to multiples of tolTime (this also rejects NaN)
        if not (abs(tSave - pre_time) <= tolTime and abs(tEnd - (pre_time + sim_time)) <= tolTime and tEnd > tSave):
            self.abort(f"1d solver :: Error: invalid save time {tSave} or end time {tEnd} from the coupler "
                       f"(pre_time = {pre_time}, pre_time + sim_time = {pre_time + sim_time})")
        # PETSc scheme: the same in the coupler configuration file, in the 0d solver and here
        def scheme_name(code):
            if code == PETSC_SCHEME_NONE:
                return "none (the ODE solver is not PETSc)"
            return next((name for name, c in PETSC_SCHEME_CODES.items() if c == code), f"unknown (code {code})")
        mine_scheme = PETSC_SCHEME_NONE if petsc_solver is None else PETSC_SCHEME_CODES[petsc_solver]
        if int(cfg[8]) != mine_scheme:
            self.abort(f"1d solver :: ERROR: PETSc scheme mismatch: the coupler configuration file uses "
                       f"{scheme_name(int(cfg[8]))}, but this 1d solver uses {scheme_name(mine_scheme)} "
                       "(PETSc_solver in [numerics] of input.ini): regenerate the model, so that the coupler configuration "
                       "file (ODEsolver, PETSc_solver), the 0d solver and the 1d solver use the same scheme "
                       "(petsc_solver in solver_info of the user inputs)")
        if int(cfg[9]) != int(N1d0dTot):
            self.abort(f"1d solver :: ERROR: the connections file of the coupler has {int(cfg[9])} 1d-0d connections, but the "
                       f"1d copy has {N1d0dTot}: regenerate the 1d input files (generate_1d: True in the user inputs)")
        self.N1d0d = int(N1d0d)
        self.N1d0dGlobal = int(N1d0dGlobal)
        return tolTime, tSave, tEnd

    def check_connections(self, pipe_1d_0d_info, json_path):
        """
        Receive the 1d-0d connections of the coupler's connections file (written with the 0d model), right after the
        handshake, and check that the fields used by the 1d solver (CONN_FIELDS_1D) are the same in its own copy
        json_path (pipe_1d_0d_info): otherwise the 1d solver would exchange the wrong values with the 0d solver.
        """
        n_conn = len(pipe_1d_0d_info)
        status = self.MPI.Status()
        self.world.Probe(source=self.coupler, tag=TAG_CONN, status=status)
        n = status.Get_count(self.MPI.DOUBLE)
        if n != n_conn*len(CONN_FIELDS):
            self.abort(f"1d solver :: ERROR: the coupler sent {n} values for the 1d-0d connections instead of "
                       f"{n_conn*len(CONN_FIELDS)} ({n_conn} connections x {len(CONN_FIELDS)} fields)")
        table = np.empty(n, dtype=np.float64)
        self.world.Recv(table, source=self.coupler, tag=TAG_CONN)
        table = table.reshape(n_conn, len(CONN_FIELDS))
        for i in range(n_conn):
            conn = pipe_1d_0d_info.get(str(i+1))
            if conn is None:
                self.abort(f"1d solver :: ERROR: connection \"{i+1}\" is missing in {json_path}: regenerate the 1d input files")
            for field in CONN_FIELDS_1D:
                value_coupler = table[i, CONN_FIELDS.index(field)]
                value_1d = conn.get(field, -1)
                if value_coupler != value_1d:
                    self.abort(f"1d solver :: ERROR: 1d-0d connection {i+1}, {field}: the connections file of the 0d model "
                               f"(read by the coupler) has {int(value_coupler)}, but the 1d copy {json_path} has {value_1d} "
                               "(-1: no such field): the 1d input files come from another generation, regenerate them "
                               "(generate_1d: True in the user inputs)")

    def recv_step_dt(self):
        """Receive {time, time step} proposed by the 0d solver at the start of a global time step."""
        buf = np.empty(DATA_LENGTH, dtype=np.float64)
        self.world.Recv(buf, source=self.coupler, tag=TAG_DT_STEP)
        return buf

    def send_step_dt(self, data):
        """Send the agreed {time, time step} at the start of a global time step."""
        self._send(data, DATA_LENGTH, TAG_DT_STEP)

    def recv_dt(self):
        """Receive {time, time step} within a global time step (internal 0d time step, or dt<0 at its end)."""
        buf = np.empty(DATA_LENGTH, dtype=np.float64)
        self.world.Recv(buf, source=self.coupler, tag=TAG_DT)
        return buf

    def send_dt(self, data):
        """Send {time, time step} within a global time step (echo of the end-of-step message)."""
        self._send(data, DATA_LENGTH, TAG_DT)

    def recv_data(self):
        """Receive the coupling data of all one-to-one 1d-0d connections: array (N1d0d, DATA_LENGTH)."""
        buf = np.empty((self.N1d0d, DATA_LENGTH), dtype=np.float64)
        if self.N1d0d > 0:
            self.world.Recv(buf, source=self.coupler, tag=TAG_DATA)
        return buf

    def send_data(self, data):
        """Send the coupling data of all one-to-one 1d-0d connections, array (N1d0d, DATA_LENGTH)."""
        if self.N1d0d > 0:
            self._send(data, self.N1d0d * DATA_LENGTH, TAG_DATA)

    def send_vol(self, data):
        """
        Send the values of the global connections (e.g. volume sum) to the coupler: array (N1d0dGlobal, DATA_LENGTH),
        one row per global connection in the order of the connections file (see connection_type()).
        """
        self._send(data, self.N1d0dGlobal * DATA_LENGTH, TAG_VOL)

    def finalize(self, time):
        """Tell the coupler that the 1d solver has completed its time loop, then finalise MPI."""
        global _active_coupling
        self._send([time, 0.0], DATA_LENGTH, TAG_DONE)
        self.local.Free()
        _active_coupling = None
        self.MPI.Finalize()

    def abort(self, message=None, code=1):
        """Print message and abort the whole MPI job (coupler and 0d solver included)."""
        if message:
            print(message, file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        self.world.Abort(code)

    def _send(self, data, length, tag):
        buf = np.ascontiguousarray(data, dtype=np.float64).reshape(-1)
        if buf.size != length:
            self.abort(f"1d solver :: Error: message with tag {tag} has {buf.size} values instead of {length}")
        self.world.Send(buf, dest=self.coupler, tag=tag)


def abort_if_active(code=1):
    """
    Abort the MPI job if a coupling is active and not finalised yet.
    Otherwise, the coupler and the 0d solver would instead wait forever.

    When called from an 'except' block, the error being handled is printed first, with its traceback
    (for sys.exit("message"), the traceback ends with "SystemExit: message"). MPI_Abort ends the
    process at once, so Python would never get to print it.
    """
    if _active_coupling is not None:
        if sys.exc_info()[0] is not None:
            print("1d solver :: Error: the 1d solver stopped with:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        _active_coupling.abort("1d solver :: Error: terminated before the end of the coupled simulation, "
                               "aborting the MPI job", code)
