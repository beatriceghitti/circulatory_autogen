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

import sys
import numpy as np

# Must match coupler_mpi_common.h
COLOR_COUPLER, COLOR_SOLVER0D, COLOR_SOLVER1D = 0, 1, 2
TAG_CONFIG, TAG_DONE, TAG_DT, TAG_DT_STEP, TAG_DATA, TAG_VOL = 1, 2, 10, 11, 20, 30
DATA_LENGTH = 2     # doubles per time/time step message and per 1d-0d connection
CONFIG_LENGTH = 5   # doubles in the TAG_CONFIG handshake: {N1d0d, couple_volume_sum, T0, nCC, tolTime}

# The coupling currently in use (if any), so that errors can abort the whole MPI job.
_active_coupling = None


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
        print(f"1d solver :: MPI coupling initialised, coupler is world rank {self.coupler}", flush=True)

    def handshake(self, N1d0d, couple_volume_sum, T0, nCC):
        """
        Receive the coupling layout from the coupler and check that it matches ours.

        Returns the time tolerance of the coupler (COUPLING_TOL_TIME in coupler_mpi_common.h),
        which the 1d solver must use in a coupled run.
        """
        cfg = np.empty(CONFIG_LENGTH, dtype=np.float64)
        self.world.Recv(cfg, source=self.coupler, tag=TAG_CONFIG)
        mine = np.array([N1d0d, 1.0 if couple_volume_sum else 0.0, T0, nCC], dtype=np.float64)
        if not np.allclose(cfg[:4], mine, rtol=1e-12, atol=0.0):
            self.abort("1d solver :: Error: coupling mismatch with the coupler "
                       f"[N1d0d, couple_volume_sum, T0, nCC]: coupler {cfg[:4].tolist()} vs 1d solver {mine.tolist()}")
        tolTime = float(cfg[4])
        if not tolTime > 0.0:
            self.abort(f"1d solver :: Error: invalid time tolerance from the coupler: {tolTime}")
        self.N1d0d = int(N1d0d)
        return tolTime

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
        """Send {1d volume sum, 0} to the coupler."""
        self._send(data, DATA_LENGTH, TAG_VOL)

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
    """
    if _active_coupling is not None:
        _active_coupling.abort("1d solver :: Error: terminated before the end of the coupled simulation, "
                               "aborting the MPI job", code)
