'''
Created on 23/05/2023

@author: Finbar Argus
'''

import math
import shutil
import numpy as np
import re
import pandas as pd
import os
import sys
import json
from sys import exit
generators_dir_path = os.path.dirname(os.path.abspath(__file__)) # absolute: written into the generated run files
root_dir = os.path.join(generators_dir_path, '../..')
sys.path.append(os.path.join(root_dir, 'src'))
from generators.CVSCellMLGenerator import CVS0DCellMLGenerator
from parsers.PrimitiveParsers import CSVFileParser
from generators.Python1DModelFilesGenerator import generate1DPythonModelFiles, generate1DPythonSimInitFile
site_pkg_path = [p for p in sys.path if 'site-packages' in p][0]
opencor_root = site_pkg_path.split('/Python/')[0]

LIBCELLML_available = True
try:
    from libcellml import Annotator, Analyser, AnalyserModel, AnalyserExternalVariable, AnalyserVariable, Generator, GeneratorProfile
    import utilities.libcellml_helper_funcs as cellml
    import utilities.libcellml_utilities as libcellml_utils
except ImportError as e:
    print("Error -> ", e)
    print('continuing without LibCellML, Warning code checks will not be available.'
          'You will need to open generated models in OpenCOR to check for errors.')
    LIBCELLML_available = False

from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import DCTERMS
from urllib.parse import urlparse

TOL_TIME = 1e-11 # 1e-13; time tolerance of fully 0d models only (coupled models use COUPLING_TOL_TIME from coupler_mpi_common.h)
REL_TOL = 1e-7
ABS_TOL = 1e-9


def round_half_away(x):
    '''
    Round to the nearest integer, halves away from zero, as std::round in C++ (Python's round() rounds halves to
    even: round(10.5) = 10, std::round(10.5) = 11). Same as _round_half_away() in auxMPIFunctions.py of the Python 1d
    solver (run1DBFmodel.py).
    '''
    a = abs(float(x))
    q = math.floor(a)  # a - q is exact in floating point
    if a - q >= 0.5:
        q += 1
    return math.copysign(float(q), x)


def get_time_window(pre_time, sim_time, dt): # OK
    '''
    Time window of a cpp model, from pre_time and sim_time of the user inputs (yaml file), as for the other model
    types: the model runs from 0 to pre_time + sim_time, and its results are saved from pre_time (every dt).
    pre_time and sim_time must be multiples of dt (the output time step), so that the first output time is
    pre_time and the last one is pre_time + sim_time.
    Returns pre_time and sim_time as floats; exits with an error if they are not valid.
    In a coupled 1d-0d model, they are written in the coupler configuration file (coupler_config.json), from which
    the coupler, the 0d solver and the 1d solver all take them (see coupling_end_time() in coupler_mpi_common.h).
    '''
    def to_float(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            print(f"get_time_window :: ERROR: {name} = {value!r} in the user inputs: cpp models need a number "
                  "(they run from 0 to pre_time + sim_time, and save their results from pre_time)")
            exit(1)
        return value
    
    pre_time = to_float('pre_time', pre_time)
    sim_time = to_float('sim_time', sim_time)
    dt = to_float('dt', dt)
    
    if not (np.isfinite(pre_time) and pre_time >= 0.0):
        print(f"get_time_window :: WARNING: pre_time >= 0 is required in the user inputs, setting pre_time = 20.0 "
              f"(pre_time = {pre_time} given)")
        pre_time = 20.0

    if not (np.isfinite(sim_time) and sim_time > 0.0):
        print(f"get_time_window :: WARNING: sim_time > 0 is required in the user inputs, setting sim_time = 2.0 "
                f"(sim_time = {sim_time} given)")
        sim_time = 2.0
    
    if not (np.isfinite(dt) and dt > 0.0):
        print(f"get_time_window :: WARNING: dt (sampling time step) must be > 0 in the user inputs, setting dt = 1e-3 "
                f"(dt = {dt} given)")
        dt = 1e-3
    
    # for name, value in (('pre_time', pre_time), ('sim_time', sim_time)):
    #     if not abs(value - round(value/dt)*dt) <= TOL_TIME:
    #         print(f"get_time_window :: ERROR: {name} = {value} must be a multiple of dt = {dt} (sampling time step) "
    #               "in the user inputs, so that the results are saved exactly from pre_time to pre_time + sim_time")
    #         exit(1)
    # rounded to the nearest multiple of dt, halves away from zero as std::round in the cpp solvers
    # (e.g. 0.0105 -> 0.011 with dt = 0.001; Python's round() would give 0.010)
    if not abs(pre_time - round_half_away(pre_time/dt)*dt) <= TOL_TIME:
        pre_time = round_half_away(pre_time/dt)*dt
    if not abs(sim_time - round_half_away(sim_time/dt)*dt) <= TOL_TIME:
        sim_time = round_half_away(sim_time/dt)*dt

    return pre_time, sim_time, dt


def get_cardiac_period(parameters_csv_abs_path): # OK
    '''
    Cardiac period T0 of a coupled 1d-0d model: the parameter T (in second) of the parameters csv file, read once at
    generation, so that the 0d model (where T is a constant) and the 1d solver use the same value. The 1d solver uses
    it as the period of its prescribed boundary conditions (e.g. the inflow): it is written in the coupler
    configuration file (coupler_config.json) and in the 1d input file (input.ini).
    It does not set the time window of the simulation (see get_time_window()): the number of cardiac cycles
    simulated is (pre_time + sim_time)/T0.
    Exits with an error if T is missing or not valid.
    '''
    def error(message):
        print(f"get_cardiac_period :: ERROR: {message}")
        exit(1)
    params_df = pd.read_csv(parameters_csv_abs_path, dtype=str, na_filter=False, skipinitialspace=True)
    params_df = params_df.rename(columns=lambda x: x.strip())
    rows = params_df[params_df['variable_name'].str.strip() == 'T']
    if len(rows) == 0:
        error(f"the parameter T (cardiac period, in second) is missing in {parameters_csv_abs_path}: "
              "a coupled 1d-0d model needs it for the 1d solver")
    if len(rows) > 1:
        error(f"the parameter T is defined {len(rows)} times in {parameters_csv_abs_path}")
    units = rows['units'].iloc[0].strip()
    if units != 'second':
        error(f"the parameter T (cardiac period) must be in second, not {units}, in {parameters_csv_abs_path}")
    try:
        T0 = float(rows['value'].iloc[0])
    except ValueError:
        T0 = np.nan
    if not (np.isfinite(T0) and T0 > 0.0):
        error(f"the parameter T (cardiac period) must be a number > 0, not {rows['value'].iloc[0]!r}, "
              f"in {parameters_csv_abs_path}")
    return T0


# PETSc time integration schemes of the cpp solvers (with the PETSc ODE solver)
PETSC_SOLVERS = {'CN': 'Crank-Nicolson', 'BDF1': 'BDF order 1 (backward Euler)', 'BDF2': 'BDF order 2'}

def get_petsc_solver(petsc_solver, couple_to_1d):
    '''
    PETSc time integration scheme of a cpp model: petsc_solver in solver_info of the user inputs (yaml file), one of
    PETSC_SOLVERS, or by default CN for a coupled 1d-0d model and BDF2 for a fully 0d model.
    The 0d solver is built with it (#define SOLVER_<scheme>). In a coupled model, it is also written in the coupler
    configuration file (PETSc_solver) and in the 1d input file (PETSc_solver in [numerics] of input.ini), and the
    coupler checks at start-up that both solvers use it (see CouplingPetscScheme in coupler_mpi_common.h).
    Exits with an error if it is not valid.
    '''
    if petsc_solver is None:
        return 'CN' if couple_to_1d else 'BDF2'
    if petsc_solver not in PETSC_SOLVERS:
        print(f"get_petsc_solver :: ERROR: petsc_solver = {petsc_solver!r} in solver_info of the user inputs: "
              f"use one of {', '.join(PETSC_SOLVERS)}")
        exit(1)
    return petsc_solver

# Functions of the generated code whose parameters are the same for every model, but whose body does not always use all
# of them (e.g. getExternalVariablesVolume() in a model without global connection, or TSRHSfunc(), which does not need
# the TS): their parameters are marked [[maybe_unused]] in their definition, so that -Wextra does not warn about them.
MAYBE_UNUSED_PARAMETERS_FUNCTIONS = ('Model0d::initialiseVariables', 'Model0d::computeComputedConstants',
                                     'Model0d::computeNonExternalVariables', 'Model0d::computeCouplingDependentVariables',
                                     'Model0d::computeCouplingSentValues', 'Model0d::getDelayVariables',
                                     'Model0d::getExternalVariables', 'Model0d::getExternalVariablesVolume', 'TSRHSfunc')

def mark_parameters_maybe_unused(cc_path):
    '''
    Marks [[maybe_unused]] the parameters of the definitions of MAYBE_UNUSED_PARAMETERS_FUNCTIONS in the generated file
    cc_path (their definitions are on one line, e.g. "void Model0d::getExternalVariablesVolume(double *varLoc)").
    '''
    with open(cc_path) as f:
        lines = f.read().split('\n')
    for iL, line in enumerate(lines):
        for name in MAYBE_UNUSED_PARAMETERS_FUNCTIONS:
            match = re.match(r'^(\w[\w\s\*]*\s+' + re.escape(name) + r'\()(.*)(\)\s*\{?\s*)$', line)
            if match and match.group(2).strip():
                parameters = [p.strip() for p in match.group(2).split(',')]
                parameters = [p if p.startswith('[[maybe_unused]]') else '[[maybe_unused]] ' + p for p in parameters]
                lines[iL] = match.group(1) + ', '.join(parameters) + match.group(3)
    with open(cc_path, 'w') as f:
        f.write('\n'.join(lines))

def global_coupling_connections(conn_1d_0d_info):
    '''
    Global 1d-0d connections of a coupled model (connections with the whole 1d model, vess1d_idx = -1, e.g. the volume
    sum), as a list of (key, connection) in the order of conn_1d_0d_info. Their values are received from the 1d solver
    in one TAG_VOL message, one row per global connection (see "Global connections" in coupler_mpi_common.h).
    Same rules as the coupler (coupling_connection_type()) and the 1d solver (connection_type() in auxMPIFunctions.py):
    the only type of global connection is the volume sum (port_volume_sum = 1), and the global connections come after
    all the one-to-one connections. Exits with an error otherwise.
    '''
    global_conns = []
    for i in range(len(conn_1d_0d_info)):
        key = str(i+1)
        conn = conn_1d_0d_info[key]
        volume_sum = conn.get("port_volume_sum", -1) == 1
        if conn["vess1d_idx"] >= 0:
            if volume_sum:
                print(f"generate_cpp :: ERROR: 1d-0d connection {key}: a connection with a 1d vessel (vess1d_idx >= 0) "
                      "cannot be a volume sum connection")
                exit(1)
            if global_conns:
                print(f"generate_cpp :: ERROR: 1d-0d connection {key} is a one-to-one connection after a global connection: "
                      "the global connections (e.g. volume sum) must come last")
                exit(1)
        elif volume_sum:
            global_conns.append((key, conn))
        # This is where more types of global connections would be added.
        else:
            print(f"generate_cpp :: ERROR: 1d-0d connection {key}: unknown type of global connection (vess1d_idx = -1): "
                  "the only type of global connection is the volume sum (port_volume_sum = 1)")
            exit(1)
    return global_conns

# A line of libCellML's C code that computes a variable: "    variables[k] = ...;" (not a comparison "==").
VARIABLE_LINE_PATTERN = re.compile(r"^\s*variables\[\d+\]\s*=(?!=)")

_VARIABLE_INDEX_PATTERN = re.compile(r"variables\[(\d+)\]")

def _split_variable_line(line2): # OK
    """
    Index of the variable computed by a line "variables[k] = ...;"
    and indices of the variables it uses.
    """
    lhs, rhs = line2.split("=", 1)
    return (int(_VARIABLE_INDEX_PATTERN.search(lhs).group(1)),
            {int(k) for k in _VARIABLE_INDEX_PATTERN.findall(rhs)})


def libcellml_variable_lines(lines): # OK
    """
    The lines "variables[k] = ...;" in computeRates() and computeVariables() of libCellML, in their order.
    """
    def body(signature):
        for iL, line2 in enumerate(lines):
            if signature in line2:
                iStart = next(i for i in range(iL+1, len(lines)) if lines[i].startswith("{"))
                iEnd = next(i for i in range(iStart, len(lines)) if lines[i].startswith("}"))
                return [l for l in lines[iStart+1:iEnd] if VARIABLE_LINE_PATTERN.search(l)]
        return []
    return body('computeRates(double '), body('computeVariables(double ')


_RATE_INDEX_PATTERN = re.compile(r"rates\[(\d+)\]")

def check_variables_use_no_rates(lines, variable_names, state_names):
    """
    The cpp solvers compute all the variables before the rates (computeRates() calls computeVariables() first), and
    the outputs with the rates array of the 0d model, which CVODE and PETSc do not update. A variable computed from a
    derivative (d(x)/dt in its equation, rates[k] in libCellML's code) would then take the rates of the previous
    evaluation, and give wrong results without any error. Currently, no module of the repository has such a variable: 
    stop the generation with an error naming them, and the variables computed from them.

    lines: the lines "variables[k] = ...;" computed at each evaluation of the rates (libCellML's code).
    variable_names, state_names: names of the variables and of the states, by index (for the message).
    """
    direct = {}
    for line2 in lines:
        rates_used = {int(k) for k in _RATE_INDEX_PATTERN.findall(line2.split("=", 1)[1])}
        if rates_used:
            direct[_split_variable_line(line2)[0]] = rates_used
    if not direct:
        return
    # the variables computed from them, directly or through other variables, take the previous rates too
    affected = set(direct)
    changed = True
    while changed:
        changed = False
        for line2 in lines:
            lhs, used = _split_variable_line(line2)
            if lhs not in affected and used & affected:
                affected.add(lhs)
                changed = True
    name = lambda k: variable_names.get(k, f"variables[{k}]")
    message = ["generate_cpp :: ERROR: these variables are computed from a derivative d(x)/dt, which the cpp solvers do "
               "not support (they compute the variables before the rates, so these variables would take the rates of "
               "the previous evaluation and give wrong results):"]
    for k in sorted(direct):
        message.append(f"    {name(k)} uses the derivative of "
                       + ", ".join(state_names.get(r, f"states[{r}]") for r in sorted(direct[k])))
    indirect = sorted(affected - set(direct))
    if indirect:
        message.append("  and these variables are computed from them: " + ", ".join(name(k) for k in indirect))
    message.append("  Write each of them from the right-hand side of the ODE of that state instead (e.g. "
                   "y = 2*(f - k*V) instead of y = 2*dV/dt, for dV/dt = f - k*V).")
    print("\n".join(message))
    exit(1)

def check_variables_computation_order(ordered_lines, all_lines, external_handled=()): # OK
    """
    Check the lines of the variables computed at each evaluation of the rates (computeNonExternalVariables() and
    computeCouplingDependentVariables()): each variable must be computed before it is used, otherwise it would be
    used with its value from the previous evaluation of the model.

    ordered_lines: the lines "variables[k] = ...;" in the order in which they are written.
    all_lines: all the lines computing variables in libCellML's code, including the external variables.
    external_handled: indices of the external variables whose use is handled: 
        - the variables received from the 1d solver (the variables using them are computed after the exchange),
        - the 1d volumes (set at the end of each time step, before the variables are computed), and
        - the delayed variables (set by getDelayVariables() before the variables are computed).

    Raises an error if a variable is used before it is computed.
    Prints a warning if a variable uses another external variable: it is set by getExternalVariables(), 
    after computeNonExternalVariables(), so these variables use its value from the previous evaluation.
    """
    computed_at = {}
    for i, line2 in enumerate(ordered_lines):
        computed_at.setdefault(_split_variable_line(line2)[0], i)
    external = {_split_variable_line(line2)[0] for line2 in all_lines if 'external' in line2} - set(external_handled)

    used_before = []
    uses_external = []
    for i, line2 in enumerate(ordered_lines):
        lhs, used = _split_variable_line(line2)
        for k in sorted(used):
            if k in computed_at and computed_at[k] >= i:
                used_before.append(f"variables[{k}] in: {line2.strip()}")
        if used & external: # take the intersection between the two sets
            uses_external.append(f"variables{sorted(used & external)} in: {line2.strip()}")
    if used_before:
        raise RuntimeError("generated computeNonExternalVariables(): variables used before they are computed "
                           "(they would take their value from the previous evaluation):\n  " + "\n  ".join(used_before))
    if uses_external:
        print("WARNING :: generated computeNonExternalVariables(): these variables use external variables, which are "
              "set after them by getExternalVariables(), so they use the external value from the previous evaluation:"
              "\n  " + "\n  ".join(uses_external))


def split_variables_at_coupling_exchange(ordered_lines, received): # OK
    """
    Split the lines of the variables computed at each evaluation of the rates around the exchange with the 1d solver
    (getExternalVariables(), which sends the values of the 1d-0d connections and then receives the 1d values).

    ordered_lines: the lines "variables[k] = ...;", each variable after those it uses.
    received: indices of the variables received from the 1d solver.

    Returns (lines_before, lines_after, depends_on):
    - lines_before: the lines of the variables that DO NOT depend on the received variables, computed before the
      exchange (computeNonExternalVariables());
    - lines_after: the lines of those that DO depend on the received variables, directly or through other variables, 
      computed after it (computeCouplingDependentVariables()).
      If computed before it, they would use the values of the previous exchange;
    - depends_on: variable of lines_after -> the received variables it depends on.
    Both lists keep the order of ordered_lines.
    """
    received = set(received)
    depends_on = {}
    lines_before = []
    lines_after = []
    for line2 in ordered_lines:
        lhs, used = _split_variable_line(line2)
        dependencies = used & received
        for k in used:
            dependencies |= depends_on.get(k, set())
        if dependencies:
            depends_on[lhs] = dependencies
            lines_after.append(line2)
        else:
            lines_before.append(line2)
    return lines_before, lines_after, depends_on


def _c_expression_to_sympy(expression): # OK
    """
    sympy expression of the right-hand side of a line of libCellML's C code, with the symbols V_k, S_k and D_k for
    variables[k], states[k] and rates[k]. Raises ValueError for the code sympy cannot read (e.g. conditional expressions).
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr
    if any(op in expression for op in ('?', '&&', '||', '==', '!=', '<', '>')):
        raise ValueError("conditional expression")
    text = re.sub(r"\b(variables|states|rates)\[(\d+)\]",
                  lambda m: {'variables': 'V', 'states': 'S', 'rates': 'D'}[m.group(1)] + '_' + m.group(2),
                  expression.strip().rstrip(';'))
    functions = {'pow': sp.Pow, 'exp': sp.exp, 'log': sp.log, 'log10': lambda x: sp.log(x, 10), 'sqrt': sp.sqrt,
                 'fabs': sp.Abs, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'sinh': sp.sinh, 'cosh': sp.cosh,
                 'tanh': sp.tanh, 'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan, 'floor': sp.floor,
                 'ceil': sp.ceiling, 'min': sp.Min, 'max': sp.Max, 'fmin': sp.Min, 'fmax': sp.Max,
                 'voi': sp.Symbol('voi')}
    try:
        return parse_expr(text, local_dict=functions)
    except Exception as e:
        raise ValueError(f"{type(e).__name__}: {e}")


def analyse_coupling_connections(connections, lines_after, depends_on, variable_names):
    """
    Decide how each 1d-0d connection exchanges its values with the 1d solver, and write the C code that computes the
    values it sends.

    The 1d solver imposes at the boundary of the 1d vessel the relation pS = P0D + sign*R*q, between the pressure pS
    and the flow q there, with the two values P0D and R sent by the 0d model, and sends back q (if the 0d module
    receives the flow: flow bc) or pS (if it receives the pressure: pressure bc, with R = 0).
    
    For a connection whose sent variable s does not depend on its received variable r, s is sent as computed before
    the exchange (computeNonExternalVariables()): P0D = s (pressure) and R = the resistance of the FV_resistance_port
    of the module, if any, for a flow bc; s (flow) for a pressure bc.
    
    Otherwise s must be an affine function of r alone, s = s0 + c*r, with s0 and c not depending on any received
    variable (checked symbolically with sympy on libCellML's lines). The lines of s are then evaluated twice before the
    exchange, with r = 0 and r = 1, which gives s0 and c exactly, and the connection sends
    - flow bc (r = q, s = pressure): P0D = s0 and R = R_FV_resistance_port + sign*c, so that
      pS = s + sign*R_FV_resistance_port*q (pS = s without an FV_resistance_port), because
      P0D + sign*R*q = s0 + sign*R_FV_resistance_port*q + sign*sign*c*q and sign*sign = 1;
    - pressure bc (r = pS, s = q): P0D = -s0/c and R = sign/c, so that q = s; the 1d solver then treats the
      connection as a flow bc (it sends back q) and the 0d model computes r = P0D + sign*R*q (converted connection).

    connections: one dict per 1d-0d connection, in the order of the exchange, with the keys
        where: description for the messages;
        received: index of the received variable (in variables);
        sent: index of the sent variable; sent_is_state: whether it is a state (index in states);
        flow_bc: True if the module receives the flow and sends a pressure, False if it receives the pressure and
            sends a flow;
        sign: +1 if the 1d vessel is upstream of the 0d module (1d outlet), -1 if it is downstream (1d inlet);
        resistance: index of the variable of the FV_resistance_port of the module (flow bc only), or None.
    lines_after, depends_on: from split_variables_at_coupling_exchange().
    variable_names: index in variables -> name, for the messages.

    Returns (code, plans): the body of computeCouplingSentValues(), in libCellML's names (variables, states, ...),
    which sets couplingSend[2*i] = P0D (or the flow) and couplingSend[2*i+1] = R for each connection i; and one dict
    per connection with 'affine' (whether its sent value depends on its received value), 'converted' (pressure bc
    exchanged as a flow bc) and 'description' (for the messages).
    Raises an error for a configuration that cannot be coupled.
    """
    def name(k):
        return variable_names.get(k, f"variables[{k}]")

    code = ""
    plans = []
    errors = []
    # The received variables are set by the exchange, not computed by the 0d model: each connection needs its own, and
    # no connection can send one of them (as its value or as its resistance), it would send back a received value.
    received_by = {}
    for conn in connections:
        received_by.setdefault(conn['received'], []).append(conn['where'])
    for k, wheres in received_by.items():
        if len(wheres) > 1:
            errors.append(f"{' and '.join(wheres)}: these connections receive their values in the same 0d variable, "
                          f"{name(k)}.")
    for i, conn in enumerate(connections):
        received, sent, flow_bc, sign = conn['received'], conn['sent'], conn['flow_bc'], conn['sign']
        resistance = conn['resistance'] if flow_bc else None
        value, res = f"couplingSend[{2*i}]", f"couplingSend[{2*i+1}]"
        sent_code = f"states[{sent}]" if conn['sent_is_state'] else f"variables[{sent}]"
        sent_name = 'pressure' if flow_bc else 'flow'
        resistance_code = f"variables[{resistance}]" if resistance is not None else "0.0"
        if not conn['sent_is_state'] and sent in received_by:
            errors.append(f"{conn['where']}: the {sent_name} sent, {name(sent)}, is the variable received by "
                          f"{'this connection' if sent == received else ', '.join(received_by[sent])}: "
                          "it would send back a value received from the 1d solver.")
            continue
        if resistance is not None and resistance in received_by:
            errors.append(f"{conn['where']}: the resistance {name(resistance)} of the FV_resistance_port is the variable "
                          f"received by {', '.join(received_by[resistance])}: it would send back a value received "
                          "from the 1d solver.")
            continue
        if resistance is not None and resistance in depends_on:
            errors.append(f"{conn['where']}: the resistance {name(resistance)} of the FV_resistance_port depends on "
                          f"the received {', '.join(name(k) for k in sorted(depends_on[resistance]))}.")
            continue
        if conn['sent_is_state'] or sent not in depends_on:
            code += f"    {value} = {sent_code};\n    {res} = {resistance_code};\n"
            plans.append({'affine': False, 'converted': False, 'description': ''})
            continue
        others = depends_on[sent] - {received}
        if others:
            errors.append(f"{conn['where']}: the {sent_name} sent, {name(sent)}, depends on the values received by other "
                          f"connections, {', '.join(name(k) for k in sorted(others))}: the 1d solver would have to "
                          "solve these connections together, which it does not do.")
            continue
        # lines computing the sent variable from the received one, in their order
        needed = {sent}
        chain = []
        for line2 in reversed(lines_after):
            lhs, used = _split_variable_line(line2)
            if lhs in needed:
                chain.append(line2)
                needed |= {k for k in used if k in depends_on}
        chain.reverse()
        try:
            import sympy as sp
            expressions = {}
            for line2 in chain:
                lhs = _split_variable_line(line2)[0]
                expression = _c_expression_to_sympy(line2.split("=", 1)[1])
                expression = expression.subs({sp.Symbol(f"V_{k}"): e for k, e in expressions.items()
                                              if sp.Symbol(f"V_{k}") in expression.free_symbols})
                expressions[lhs] = expression
            r = sp.Symbol(f"V_{received}")
            coefficient = sp.diff(expressions[sent], r)
            if coefficient.has(r):
                coefficient = sp.simplify(coefficient)
        except ValueError as e:
            errors.append(f"{conn['where']}: the {sent_name} sent, {name(sent)}, depends on the received {name(received)}, "
                          f"and its lines could not be analysed ({e}):\n    " + "\n    ".join(l.strip() for l in chain))
            continue
        if coefficient.has(r):
            errors.append(f"{conn['where']}: the {sent_name} sent, {name(sent)}, depends nonlinearly on the received "
                          f"{name(received)}: the 1d solver can only impose a linear relation between pressure and flow.")
            continue
        readable = coefficient.subs({sp.Symbol(f"V_{k}"): sp.Symbol(n) for k, n in variable_names.items()
                                     if sp.Symbol(f"V_{k}") in coefficient.free_symbols})
        chain_code = "".join("    " + line2 + "\n" for line2 in chain)
        code += "    {\n"
        code += f"        double received = variables[{received}];\n"
        code += f"        variables[{received}] = 0.0;\n" + chain_code
        code += f"        double value0 = variables[{sent}];\n"
        code += f"        variables[{received}] = 1.0;\n" + chain_code
        code += f"        double coefficient = variables[{sent}]-value0;\n"
        code += f"        variables[{received}] = received;\n"
        # the received variable cancels out: the value sent does not depend on it (computed after the exchange anyway)
        cancels = sp.simplify(coefficient) == 0
        if flow_bc:
            code += f"        {value} = value0;\n"
            code += f"        {res} = {resistance_code}{'+' if sign > 0 else '-'}coefficient;\n"
        elif cancels:
            code += f"        {value} = value0;\n"
            code += f"        {res} = 0.0;\n"
        else:
            code += f"        {value} = -value0/coefficient;\n"
            code += f"        {res} = {'1.0' if sign > 0 else '-1.0'}/coefficient;\n"
        code += "    }\n"
        if flow_bc:
            description = (f"the pressure sent, {name(sent)}, depends linearly on the received flow {name(received)} "
                           f"(coefficient {readable}): sent as P0D = {name(sent)} at zero flow and "
                           f"R = {'' if resistance is None else name(resistance) + ' + '}{'' if sign > 0 else '-'}coefficient")
        else:
            description = (f"the flow sent, {name(sent)}, depends linearly on the received pressure {name(received)} "
                           f"(coefficient {readable}): exchanged as a flow bc, with P0D = -{name(sent)}(0)/coefficient "
                           f"and R = {'' if sign > 0 else '-'}1/coefficient; {name(received)} = P0D {'+' if sign > 0 else '-'} R*q")
        plans.append({'affine': True, 'converted': not flow_bc and not cancels, 'description': description})
    if errors:
        raise RuntimeError("1d-0d coupling: configurations that cannot be coupled:\n  " + "\n  ".join(errors))
    return code, plans

class CVS0DCppGenerator(object):
    '''
    Generates Cpp files from CellML files and annotations for a 0D model

    # TODO This code is not good. it needs a complete re-write. 
    # The class could be made much much simpler. Don't try to 
    # work with this without talking to Finbar. Make him simplify it before you
    # work with this.
    '''


    def __init__(self, model, generated_model_subdir, file_prefix, resources_dir=None, 
                 solver='CVODE', dtSample=1e-3, dtSolver=1e-4, nMaxSteps=5000,
                 pre_time=None, sim_time=None, T0=None, petsc_solver=None,
                 couple_to_1d=False, cpp_generated_models_dir=None,
                 model_1d_config_path=None, solver_1d_path=None, create_main_0d=False,
                 conn_1d_0d_info=None, results_dir=None, DEBUG=False):
        '''
        Constructor
        solver_1d_path: path of the Python 1d solver (run1DBFmodel.py, solver_1d_path in the user inputs), written into the
            coupler configuration file of a coupled model (solver1d_path)
        results_dir: folder of the results of the cpp models (cpp_results_dir in the user inputs): the results of this
            model are saved in <results_dir>/<file_prefix>/. By default, simulation_outputs_cpp next to the generated
            models folder (the parent folder of generated_model_subdir).
        '''
        if LIBCELLML_available == False:
            print("CVS0DCppGenerator __init__ :: ERROR: LibCellML is not available, cannot generate Cpp files.")
            exit(1)
            
        self.model = model
        self.generated_model_subdir = generated_model_subdir
        if not os.path.exists(self.generated_model_subdir):
            os.mkdir(self.generated_model_subdir)
        self.file_prefix = file_prefix
        self.file_prefix_with_ids = f'{self.file_prefix}-with-ids'
        if resources_dir is None:
            self.resources_dir = os.path.join(generators_dir_path, '../../resources')
        else:
            self.resources_dir = resources_dir 
        self.generated_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.file_prefix + '.cellml')
        self.generated_wID_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.file_prefix_with_ids + '.cellml')
        self.annotated_model_file_path = os.path.join(self.generated_model_subdir, 
                                                      self.file_prefix_with_ids + '--annotations.ttl')
        
        self.cellml_model = None 
        self.couple_to_1d = couple_to_1d
        self.couple_volume_sum = False
        if self.couple_to_1d:
            self.output_cpp_file_name = "model0d" # always the same, independent of model name, 
                                                    # to allow for coupling to cpp 1d model.
            self.conn_1d_0d_info = conn_1d_0d_info
        else:
            self.output_cpp_file_name = self.file_prefix
            self.conn_1d_0d_info = None

        self.external_headers = []
        
        if self.couple_to_1d:
            # absolute paths: they are written into coupler_config.json and the run script, which run from other folders
            self.model_1d_config_path = os.path.abspath(model_1d_config_path) if model_1d_config_path else model_1d_config_path
            self.solver_1d_path = os.path.abspath(solver_1d_path) if solver_1d_path else solver_1d_path
            self.create_main = create_main_0d
            # self.external_headers += ['model1d.h']
        else:
            self.create_main = True
        
        if cpp_generated_models_dir is None:
            if self.couple_to_1d:
                self.cpp_generated_models_dir = self.generated_model_subdir + "_cpp"
            else:
                self.cpp_generated_models_dir = self.generated_model_subdir
        else:
            self.cpp_generated_models_dir = cpp_generated_models_dir
            # (the name of the cpp folder only, not of its parent folders)
            if self.couple_to_1d and 'fvm' in os.path.basename(os.path.normpath(cpp_generated_models_dir)):
                self.external_headers += ['model1d.h']
        # absolute path: written into coupler_config.json, the run script and main0d.cpp (results folder), which are
        # used from other folders (the run script changes directory)
        self.cpp_generated_models_dir = os.path.abspath(self.cpp_generated_models_dir)

        if not os.path.exists(self.cpp_generated_models_dir):
            # os.mkdir(self.cpp_generated_models_dir)
            os.makedirs(self.cpp_generated_models_dir, exist_ok=True)

        # results folder of this model, <results_dir>/<file_prefix>/ (absolute: written into the generated main, which is
        # run from other folders), the same for fully 0d and coupled models
        if results_dir is None:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(self.generated_model_subdir))),
                                       'simulation_outputs_cpp')
        self.results_dir = os.path.join(os.path.abspath(results_dir), self.file_prefix)
        
        self.solver = solver
        # PETSc time integration scheme (with the PETSc ODE solver), see get_petsc_solver()
        self.petsc_solver = get_petsc_solver(petsc_solver, couple_to_1d)
        if self.couple_to_1d and self.solver == 'PETSC' and self.petsc_solver == 'BDF2':
            print("CVS0DCppGenerator __init__ :: WARNING: the 1d solver does not implement the 1d-0d coupling with BDF2 yet: "
                  "the coupled simulation will stop at its first time step (use CN or BDF1 for petsc_solver in solver_info "
                  "of the user inputs)")
        self.dtSample = dtSample
        self.dtSolver = dtSolver
        self.nMaxSteps = nMaxSteps
        # time window (pre_time and sim_time of the user inputs, see get_time_window()) and, for a coupled model,
        # cardiac period (parameter T, see get_cardiac_period())
        self.pre_time, self.sim_time, self.dtSample = get_time_window(pre_time, sim_time, dtSample)
        self.T0 = T0
        if self.couple_to_1d and not (T0 is not None and np.isfinite(T0) and T0 > 0.0):
            print(f"CVS0DCppGenerator __init__ :: ERROR: a coupled 1d-0d model needs the cardiac period T0 > 0 "
                  f"(parameter T, see get_cardiac_period()), not {T0}")
            exit(1)

        # these are for delayed variables
        self.variables_to_delay = []
        self.delayed_variables = []
        self.delay_amounts = []
        self.independent_variables = []
        # these are for two-way coupling
        self.port_input_variables = []
        self.port_output_variables = []
        self.control_variables = [] # This is for extra control_variables, e.g. resistance in a 1D FV model
        self.connection_vessel_indices = []
        self.connection_vessel_types = []
        self.connection_vessel_inlet_or_outlet = []
        self.connection_vessel_flow_or_pressure_bc = []
        # these are for one-way coupling
        self.input_variables = []
        self.output_variables = []
        # this is for volume sum one-way coupling
        self.volume_sum_variables = []
        self.volume_sum_indeces = []
        self.volume_sum_types = []

        self.DEBUG = DEBUG

        # TODO should the annotations stuff be in a different class? It's not crucial for 
        # running of the model but could be useful for making sure we have annotated models
        # with the delays, coupling variables etc.

        # define some URIs for things we need

        # use this URI to identify delayed variables - not the perfect URI, but will do for now
        #     This is actually "delayed differential equation model" from the MAMO ontology
        #delay_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000089')
        self.delay_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-variable')
        self.variable_to_delay_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
        self.independent_variable_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#variable-to-delay')
        self.delay_amount_uri = URIRef('https://github.com/nickerso/libcellml-python-utils/properties.rst#delay-amount')

        # use this for some random thing that I want to define - http://example.com is a good base for things that will never resolve
        self.stuff_uri = URIRef('http://example.com/cool-thing#21')

        # a "readout" variable that we maybe want to connect to something external?
        self.timecourse_readout_uri = URIRef('http://identifiers.org/mamo/MAMO_0000031')
        self.output_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000018')
        self.input_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000017')
        # TODO this is randomly chosen for now, update this
        self.external_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_9239430432')
        self.control_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_9239430433')
        # TODO this is randomly chosen for now, update this
        self.volume_sum_variable_uri = URIRef('http://identifiers.org/mamo/MAMO_0000060')

        self.g = Graph()
        
    def init_from_inp_data_dict(self, inp_data_dict):
        '''
        Initialise the generator from an input data dictionary
        '''
        raise NotImplementedError("Not implemented yet")

    def __append_1d_coupling(self, vessel_tup, side, vessel_1d): # OK
        '''
        Find the variables of the 0d module vessel_tup that are exchanged with the 1d (FV) vessel vessel_1d, connected
        to one of its vessel_ports in its entrance_ports (side 'entrance') or exit_ports (side 'exit'), and append them
        to the lists of the 1d-0d connections.

        Flow bc (the 0d module receives the flow from the 1d vessel): the module sends the pressure of the vessel_port,
        and the resistance of its FV_resistance_port if it has one (a resistance applied by the 1d solver only).
        Pressure bc (the module receives the pressure): the module sends the flow of the vessel_port, and no resistance.
        In each list of ports, the k-th FV_resistance_port belongs to the k-th vessel_port.

        Exits with an error for a configuration that cannot be coupled. How the variables sent depend on the variables
        received is analysed later, in generate_cpp(), once libCellML has analysed the model
        (see analyse_coupling_connections()).
        '''
        vessels_df = self.model.vessels_df
        module = f"0d module {vessel_tup.name} (module_type {vessel_tup.module_type}, BC_type {vessel_tup.BC_type})"

        def error(message):
            print(f"__append_1d_coupling :: ERROR: coupling of the 1d vessel {vessel_1d} to the {side} of the {module}: {message}")
            exit(1)

        def vessel_type_of(name):
            return vessels_df.loc[vessels_df["name"] == name, "vessel_type"].iloc[0]

        def is_multi_port(port):
            return port.get("multi_port") in (True, 'True')

        ports = vessel_tup.entrance_ports if side == 'entrance' else vessel_tup.exit_ports
        neighbours = list(vessel_tup.inp_vessels if side == 'entrance' else vessel_tup.out_vessels)
        vessel_ports = [port for port in ports if port["port_type"] == "vessel_port"]
        resistance_ports = [port for port in ports if port["port_type"] == "FV_resistance_port"]

        # which vessel_port of the module is connected to the 1d vessel
        if len(vessel_ports) == 0:
            error(f"the module has no vessel_port in its {side}_ports.")
        
        if vessel_tup.name == 'heart':
            # the vessel_port is found from the name of the 1d vessel
            if side == 'exit':
                def matches(port):
                    return ((('aorta' in vessel_1d or 'aortic_root' in vessel_1d) and port["variables"][1] == 'u_root')
                            or ('par' in vessel_1d and port["variables"][1] == 'u_par'))
            else:
                def matches(port):
                    return (('ivc' in vessel_1d and port["variables"][0] == 'v_ivc')
                            or ('svc' in vessel_1d and port["variables"][0] == 'v_svc')
                            or ('pvn' in vessel_1d and port["variables"][0] == 'v_pvn'))
            found = [k for k, port in enumerate(vessel_ports) if matches(port)]
            if len(found) != 1:
                error(f"{len(found)} vessel_ports of the heart match the name of the 1d vessel, instead of one "
                      "(exit: 'aorta' or 'aortic_root' -> u_root, 'par' -> u_par; entrance: 'ivc', 'svc', 'pvn').")
            k = found[0]
        elif any(is_multi_port(port) for port in vessel_ports):
            #XXX TODO Extend: For now, coupling generator refuses to couple a 1D vessel to a multi-port side
            error("its vessel_port is a multi_port (junction module): coupling it to a 1d vessel is not supported.")
        elif len(vessel_ports) == 1:
            k = 0
        elif side == 'exit':
            # Several exit vessel_ports (e.g. split junction: v_out_1, v_out_2): the 1d vessel must get the one that
            # the CellML generator (__write_module_mapping_for_row) leaves to it. The CellML generator connects them
            # while processing the row of this module: it walks out_vessels in order and gives each one the next
            # free exit vessel_port. A 1d out_vessel also takes one (no mapping is written for it, but the port is
            # marked as used), so the k-th out_vessel gets the k-th exit vessel_port: k = position in out_vessels.
            # Example: out_vessels = [z (0d), A_dn (1d)] -> z gets v_out_1, A_dn gets v_out_2 (k = 1).
            # If an out_vessel is a Min_/MinNout_ junction, the CellML generator does not mark the ports as used
            # (they are shared), so the rule does not hold.
            if any(vessel_type_of(name).startswith(("Min_", "MinNout_")) for name in neighbours):
                error("the module has several exit vessel_ports and one of its out_vessels is a Min_/MinNout_ "
                      "junction: which vessel_port is connected to the 1d vessel is not defined.")
            k = neighbours.index(vessel_1d)
        else:
            # Several entrance vessel_ports (e.g. merge junction: v_in_1, v_in_2). The CellML generator connects them
            # while processing the rows of the upstream modules: each 0d inp_vessel takes the first free entrance
            # vessel_port of this module. The rows of the 1d vessels (module_format cpp) are skipped, so they never
            # take one: the 0d inp_vessels get the first ports, and the 1d vessels the ones after them, in the order
            # of inp_vessels (the CellML generator writes nothing for them, so this order is a choice made here).
            # k = (number of 0d inp_vessels) + (position of this 1d vessel among the 1d inp_vessels).
            # Example: inp_vessels = [A_up (1d), z (0d)] -> z gets v_in_1, A_up gets v_in_2 (k = 1 + 0), which is
            # not the position of A_up in inp_vessels.
            # If an inp_vessel is a Nout_/MinNout_ junction, the CellML generator does not mark the ports as used
            # (they are shared), so the rule does not hold.
            if any(vessel_type_of(name).startswith(("Nout_", "MinNout_")) for name in neighbours):
                error("the module has several entrance vessel_ports and one of its inp_vessels is a Nout_/MinNout_ "
                      "junction: which vessel_port is connected to the 1d vessel is not defined.")
            inp_1d = [name for name in neighbours if vessel_type_of(name) == "FV1D_vessel"]
            k = (len(neighbours) - len(inp_1d)) + inp_1d.index(vessel_1d)
        
        if k >= len(vessel_ports):
            error(f"it would be connected to vessel_port {k+1}, but the module has only {len(vessel_ports)} vessel_ports "
                  f"in its {side}_ports (check {'out' if side == 'exit' else 'inp'}_vessels in the vessel array).")

        if len(resistance_ports) not in (0, len(vessel_ports)):
            error(f"its {side}_ports have {len(resistance_ports)} FV_resistance_port(s) and {len(vessel_ports)} "
                  "vessel_port(s): the k-th FV_resistance_port belongs to the k-th vessel_port, so there must be one for "
                  "each vessel_port.")

        component = self.cellml_model.component(vessel_tup.name)

        def variable_id(name, port_type):
            variable = component.variable(name)
            if variable is None:
                error(f"variable {name} of its {port_type} not found in component {vessel_tup.name} of the generated "
                      "CellML (it must be listed as 'access' in variables_and_units of the module configuration).")
            return variable.id()

        vessel_port = vessel_ports[k]
        bc = vessel_tup.BC_type[0] if side == 'entrance' else vessel_tup.BC_type[1]
        if bc == 'v': # flow bc: the module receives the flow, sends a pressure and a resistance
            flow_or_pressure_bc = "flow"
            input_id = variable_id(vessel_port["variables"][0], "vessel_port")
            output_id = variable_id(vessel_port["variables"][1], "vessel_port")
            if resistance_ports:
                control_id = variable_id(resistance_ports[k]["variables"][0], "FV_resistance_port")
            else:
                control_id = None
        elif bc == 'p': # pressure bc: the module receives the pressure, sends the flow
            flow_or_pressure_bc = "pressure"
            input_id = variable_id(vessel_port["variables"][1], "vessel_port")
            output_id = variable_id(vessel_port["variables"][0], "vessel_port")
            control_id = None
            if resistance_ports:
                print(f"WARNING :: coupling of the 1d vessel {vessel_1d} to the {side} of the {module}: the module "
                      "receives the pressure and sends its flow, so its FV_resistance_port is not used.")
        else:
            error(f"unknown BC type {vessel_tup.BC_type} for a connection to a 1d vessel (expected 'v' or 'p' "
                  f"for the {side}).")

        self.connection_vessel_inlet_or_outlet.append("inlet" if side == 'entrance' else "outlet")
        self.connection_vessel_flow_or_pressure_bc.append(flow_or_pressure_bc)
        self.port_input_variables.append(input_id)
        self.port_output_variables.append(output_id)
        self.control_variables.append(control_id)

    def annotate_cellml(self):
        print("annotating CellML files, before Cpp generation...")
        #XXX Bea: is this really necessary???
        # TODO most of this function 
        # isn't needed for the generation anymore, but is it good to keep annotation in here? Probably.
        print(self.generated_model_file_path)
        self.cellml_model = cellml.parse_model(self.generated_model_file_path, False)
        annotator = Annotator()
        annotator.setModel(self.cellml_model)

        if annotator.assignAllIds():
            print('Some entities have been assigned an ID, you should save the model!')
        else:
            print('Everything already had an ID.')

        duplicates = annotator.duplicateIds()
        if len(duplicates) > 0:
            print("There are some duplicate IDs, behaviour may be unreliable...")
            print(duplicates)

        # blow away all the IDs and reassign them
        annotator.clearAllIds()
        annotator.assignAllIds()
        print("All IDs reassigned")

        model_string = cellml.print_model(self.cellml_model)
        print("Now printing the model")
        print(model_string)
        
        # and save the updated model to a new file
        # - note, we need the model filename for making our annotations later
        with open(self.generated_wID_model_file_path, 'w') as f:
            f.write(model_string)
        print("Updated model saved to a new file")
        print('\n')
        
        # get the ID of the variables we want to annotate
        #XXX The below for loops are the only part of annotate_cellml that is needed to run generate_cpp
        print("Get the ID of the variables we want to annotate")
        for vessel_tup in self.model.vessels_df.itertuples():
            if "delay_info" in vessel_tup._fields:
                # if vessel_tup.delay_info is not "None":
                if vessel_tup.delay_info != "None":
                    for II in range(len(vessel_tup.delay_info["variables_to_delay"])):
                        variable_to_delay = vessel_tup.delay_info["variables_to_delay"][II]
                        delayed_variable = vessel_tup.delay_info["delayed_variables"][II]
                        delay_amount = vessel_tup.delay_info["delay_amounts"][II]
                        
                        self.variables_to_delay.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(variable_to_delay).id())
                        self.delayed_variables.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(delayed_variable).id())
                        self.delay_amounts.append(self.cellml_model.component(vessel_tup.name \
                                                                        ).variable(delay_amount).id())
                        # TODO make functionality to delay from other variables?
                        self.independent_variables.append(self.cellml_model.component("environment" \
                                                                        ).variable('time').id())
                        
        print('Delayed variables :: DONE')
        
        fv1d_vessel_index_list = []
        for idx0, vessel_tup in enumerate(self.model.vessels_df.itertuples()):
            if vessel_tup.vessel_type!="FV1D_vessel":
                # print(vessel_tup.name, vessel_tup.inp_vessels, vessel_tup.out_vessels)
                for out_vessel in vessel_tup.out_vessels:
                    out_vessel_row = self.model.vessels_df.loc[self.model.vessels_df["name"] == out_vessel].squeeze()
                    if out_vessel_row["vessel_type"] == "FV1D_vessel":
                        # Finbar's format: get the index from the out_vessel name which is of format "FV1D_#", or "FV1D_##", or "FV1D_###", etc
                        # # fv1d_vessel_index = re.findall(r'\d+$', out_vessel)[0]
                        # fv1d_vessel_index = re.findall(r'\d+$', out_vessel) # "\d+" -> one or more digits, "$" -> at the end of the string
                        if out_vessel.startswith('FV1D_'):
                            fv1d_vessel_index = re.findall(r'\d+$', out_vessel)
                        else:
                            fv1d_vessel_index = []
                        
                        if len(fv1d_vessel_index)==0: #XXX then, use different indexing system
                            # print("WARNING: FV1D vessel has NO idx", out_vessel, fv1d_vessel_index)
                            idx1 = -1
                            for i in range(len(self.conn_1d_0d_info)):
                                conn1d0d = self.conn_1d_0d_info[str(i+1)]
                                if (conn1d0d["vess0d_idx"] == idx0 and conn1d0d.get("vess1d_name", out_vessel) == out_vessel
                                        and conn1d0d["cellml_bc_in0_or_out1"] == 1): # this 0d module, this 1d vessel, this side
                                    idx1 = conn1d0d["vess1d_idx"]
                                    break
                            if idx1==-1:
                                print(f"annotate_cellml :: ERROR: idx of 1D FV vessel {out_vessel} connected to 0D module {vessel_tup.name} not found.")
                                exit(1)
                            fv1d_vessel_index = str(idx1)
                            fv1d_vessel_index_list.append([out_vessel, fv1d_vessel_index])
                            print("FV1D vessel idx found", out_vessel, fv1d_vessel_index)

                        elif len(fv1d_vessel_index)==1:
                            # index of the 1d vessel in the 1d vessel array, written in its name (any number of digits)
                            fv1d_vessel_index = fv1d_vessel_index[0]
                            
                            # best to do a sanity check, for compatibility between Finbar's old code and Bea's new code
                            idx1 = -1
                            for i in range(len(self.conn_1d_0d_info)):
                                conn1d0d = self.conn_1d_0d_info[str(i+1)]
                                if (conn1d0d["vess0d_idx"] == idx0 and conn1d0d.get("vess1d_name", out_vessel) == out_vessel
                                        and conn1d0d["cellml_bc_in0_or_out1"] == 1): # this 0d module, this 1d vessel, this side
                                    idx1 = conn1d0d["vess1d_idx"]
                                    break
                            if idx1==-1:
                                print(f"annotate_cellml :: ERROR: idx of 1D FV vessel {out_vessel} connected to 0D module {vessel_tup.name} not found.")
                                exit(1)
                            if int(fv1d_vessel_index) != idx1: # as numbers (e.g. FV1D_07 is index 7)
                                print(f"annotate_cellml :: ERROR: the index in the name of the 1D FV vessel {out_vessel} ({fv1d_vessel_index}) "
                                      f"is not its index in the 1D vessel array ({idx1})")
                                exit(1)
                            fv1d_vessel_index = str(idx1)
                            
                            fv1d_vessel_index_list.append([out_vessel, fv1d_vessel_index])


                        self.connection_vessel_indices.append(fv1d_vessel_index)
                        self.connection_vessel_types.append("FV_1d") # TODO only option for now, can be extended to other kinds of coupling.
                        self.__append_1d_coupling(vessel_tup, 'exit', out_vessel) #XXX
                
                for inp_vessel in vessel_tup.inp_vessels:
                    inp_vessel_row = self.model.vessels_df.loc[self.model.vessels_df["name"] == inp_vessel].squeeze()
                    if inp_vessel_row["vessel_type"] == "FV1D_vessel":
                        # # fv1d_vessel_index = re.findall(r'\d+$', inp_vessel)[0]
                        # fv1d_vessel_index = re.findall(r'\d+$', inp_vessel) # "\d+" -> one or more digits, "$" -> at the end of the string
                        if inp_vessel.startswith('FV1D_'):
                            fv1d_vessel_index = re.findall(r'\d+$', inp_vessel)
                        else:
                            fv1d_vessel_index = []
                        
                        if len(fv1d_vessel_index)==0: #XXX then, use different indexing system
                            # print("WARNING: FV1D vessel has NO idx", inp_vessel, fv1d_vessel_index)
                            idx1 = -1
                            for i in range(len(self.conn_1d_0d_info)):
                                conn1d0d = self.conn_1d_0d_info[str(i+1)]
                                if (conn1d0d["vess0d_idx"] == idx0 and conn1d0d.get("vess1d_name", inp_vessel) == inp_vessel
                                        and conn1d0d["cellml_bc_in0_or_out1"] == 0): # this 0d module, this 1d vessel, this side
                                    idx1 = conn1d0d["vess1d_idx"]
                                    break
                            if idx1==-1:
                                print(f"annotate_cellml :: ERROR: idx of 1D FV vessel {inp_vessel} connected to 0D module {vessel_tup.name} not found.")
                                exit(1)
                            fv1d_vessel_index = str(idx1)
                            fv1d_vessel_index_list.append([inp_vessel, fv1d_vessel_index])
                            print("FV1D vessel idx found", inp_vessel, fv1d_vessel_index)
                            
                        elif len(fv1d_vessel_index)==1:
                            # index of the 1d vessel in the 1d vessel array, written in its name (any number of digits)
                            fv1d_vessel_index = fv1d_vessel_index[0]

                            # best to do a sanity check, for compatibility between Finbar's old code and Bea's new code
                            idx1 = -1
                            for i in range(len(self.conn_1d_0d_info)):
                                conn1d0d = self.conn_1d_0d_info[str(i+1)]
                                if (conn1d0d["vess0d_idx"] == idx0 and conn1d0d.get("vess1d_name", inp_vessel) == inp_vessel
                                        and conn1d0d["cellml_bc_in0_or_out1"] == 0): # this 0d module, this 1d vessel, this side
                                    idx1 = conn1d0d["vess1d_idx"]
                                    break
                            if idx1==-1:
                                print(f"annotate_cellml :: ERROR: idx of 1D FV vessel {inp_vessel} connected to 0D module {vessel_tup.name} not found.")
                                exit(1)
                            if int(fv1d_vessel_index) != idx1: # as numbers (e.g. FV1D_07 is index 7)
                                print(f"annotate_cellml :: ERROR: the index in the name of the 1D FV vessel {inp_vessel} ({fv1d_vessel_index}) "
                                      f"is not its index in the 1D vessel array ({idx1})")
                                exit(1)
                            fv1d_vessel_index = str(idx1)

                            fv1d_vessel_index_list.append([inp_vessel, fv1d_vessel_index])
                        
                        self.connection_vessel_indices.append(fv1d_vessel_index)
                        self.connection_vessel_types.append("FV_1d") # TODO only option for now, can be extended to other kinds of coupling.
                        self.__append_1d_coupling(vessel_tup, 'entrance', inp_vessel) #XXX

        # each 1d-0d connection needs its own received variable (the annotations are attached to it)
        duplicated = {v for v in self.port_input_variables if self.port_input_variables.count(v) > 1}
        if duplicated:
            print(f"annotate_cellml :: ERROR: the same 0d variable would receive the values of several 1d-0d connections: {duplicated}")
            exit(1)

        # print(fv1d_vessel_index_list)
        print('1d-0d coupled variables :: DONE')


        for idx0, vessel_tup in enumerate(self.model.vessels_df.itertuples()):
            if vessel_tup.vessel_type=="FV1D_volume_sum": # TODO only option for now, can be extended to other kinds of coupling.
                self.volume_sum_variables.append(self.cellml_model.component(vessel_tup.name \
                                            ).variable('q_sum').id())
                
                idx_sum = -1
                for i in range(len(self.conn_1d_0d_info)):
                    conn1d0d = self.conn_1d_0d_info[str(i+1)]
                    if "port_volume_sum" in conn1d0d:
                        if conn1d0d["port_volume_sum"] == 1 and conn1d0d["vess0d_idx"] == idx0:
                            idx_sum = idx0
                            break
                if idx_sum==-1:
                    print(f"annotate_cellml :: ERROR: idx of 1D FV volume sum connection not found.")
                    exit(1)
                self.volume_sum_indeces.append(str(idx_sum))

                self.volume_sum_types.append("FV_1d_sum") 

        if len(self.volume_sum_variables)>0:
            self.couple_volume_sum = True
        
        print(f'Volume sum variables :: DONE. couple_volume_sum set to {self.couple_volume_sum}')
        # print(self.volume_sum_variables)
        # print(self.volume_sum_indeces)
        # print(self.volume_sum_types)
        # print('\n')
        
        
        # self.input_variables = [model.component(vessel_tup.name).variable('v_in').id()]
        # self.output_variables = [model.component(vessel_tup.name).variable('u').id()]

        # Create an RDF URI node for our variable to use as the subject for multiple triples
        # note: we are going to serialise the RDF graph into the same folder, so we need a URI that is relative to the intended file
        
        # print(self.generated_wID_model_file_path)
        # print(self.cpp_generated_models_dir)

        variable_to_delay_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                             '#' + variable) for variable in self.variables_to_delay]
        delay_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                          '#' + variable) for variable in self.delayed_variables]
        delay_amount_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                        '#' + delay_amount) for delay_amount in self.delay_amounts]
        independent_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                                '#' + variable) for variable in self.independent_variables]

        port_input_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                               '#' + variable) for variable in self.port_input_variables]
        port_output_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                                '#' + variable) for variable in self.port_output_variables]
        # control_variable_name_uri = [URIRef(self.generated_wID_model_file_path +
        #                                     '#' + variable) for variable in self.control_variables]
        control_variable_name_uri = []
        for variable in self.control_variables:
            if variable is not None:
                control_variable_name_uri.append(URIRef(self.generated_wID_model_file_path +
                                            '#' + variable))
        
        print("control_variable_name_uri:", control_variable_name_uri)

        external_variable_idx_uri = []                                        
        for idx, type, inlet_or_outlet, flow_or_pressure_bc in zip(self.connection_vessel_indices, self.connection_vessel_types, 
                             self.connection_vessel_inlet_or_outlet, self.connection_vessel_flow_or_pressure_bc):
            external_variable_idx_uri.append(URIRef(self.cpp_generated_models_dir) +
                                        '#' + type + '#' + idx + '#' + inlet_or_outlet + '#' + flow_or_pressure_bc)

        input_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                          '#' + variable) for variable in self.input_variables] # this is empty as long as above lines are commented
        output_variable_name_uri = [URIRef(self.generated_wID_model_file_path + 
                                           '#' + variable) for variable in self.output_variables] # this is empty as long as above lines are commented
        
        # volume_sum_variable_name_uri = [URIRef(self.generated_wID_model_file_path +
        #                                     '#' + variable) for variable in self.volume_sum_variables]
        # volume_sum_variable_name_uri = [URIRef(self.cpp_generated_models_dir +
        #                                     '#' + variable) for variable in self.volume_sum_variables]
        volume_sum_variable_name_uri = []                                        
        for variable, idx, type in zip(self.volume_sum_variables, self.volume_sum_indeces, self.volume_sum_types):
            volume_sum_variable_name_uri.append(URIRef(self.cpp_generated_models_dir) +
                                                '#' + type + '#' + idx + '#' + variable)
    
        
        # Add triples using store's add() method.
        # We're using the Dublin Core term "type" to associate the variable with the delay...
        for II in range(len(variable_to_delay_name_uri)):
            self.g.add((variable_to_delay_name_uri[II], DCTERMS.type, self.variable_to_delay_uri))
            self.g.add((variable_to_delay_name_uri[II], self.delay_variable_uri, delay_variable_name_uri[II]))
            self.g.add((variable_to_delay_name_uri[II], self.independent_variable_uri, independent_variable_name_uri[II]))
            self.g.add((variable_to_delay_name_uri[II], self.delay_amount_uri, delay_amount_name_uri[II]))
        # Set coupling variables
        IIc = 0
        for II in range(len(self.port_input_variables)):
            # for this input variable, add the output variable as a coupling variable
            self.g.add((port_input_variable_name_uri[II], DCTERMS.type, self.input_variable_uri))
            self.g.add((port_input_variable_name_uri[II], self.output_variable_uri, port_output_variable_name_uri[II]))
            self.g.add((port_input_variable_name_uri[II], self.external_variable_uri, external_variable_idx_uri[II]))
            if len(control_variable_name_uri) > 0:
                if self.control_variables[II] is not None:
                    self.g.add((port_input_variable_name_uri[II], self.control_variable_uri, control_variable_name_uri[IIc]))
                    IIc += 1

        for II in range(len(self.output_variables)):
            self.g.add((output_variable_name_uri[II], DCTERMS.type, self.output_variable_uri))
        for II in range(len(self.input_variables)):
            self.g.add((input_variable_name_uri[II], DCTERMS.type, self.input_variable_uri))

        for II in range(len(self.volume_sum_variables)):
            self.g.add((volume_sum_variable_name_uri[II], DCTERMS.type, self.volume_sum_variable_uri))

        
        # print all the data in the turtle format
        print(self.g.serialize(format='ttl'))

        # and save to a file
        with open(self.annotated_model_file_path, 'w') as f:
            f.write(self.g.serialize(format='ttl'))


    def generate_cellml(self, inp_data_dict):
        print("generating CellML files, before Cpp generation")
        cellml_generator = CVS0DCellMLGenerator(self.model, inp_data_dict)
        return cellml_generator.generate_files() # True if the CellML model opens in OpenCOR (or runs in Python)

    def set_annotated_model_file_path(self, annotated_model_file_path):
        # This is for if we want to use a cellml file that it already annotated, i.e, one we didn't annotate.
        # TODO TEST THIS
        self.annotated_model_file_path = annotated_model_file_path

    def generate_cpp(self):

        print("now Cpp generation can finally starts")

        # time tolerance written in the generated code: a coupled 0d model must use the same value as the coupler,
        # so it takes it from coupler_mpi_common.h (included only when coupled); a fully 0d model has no coupler,
        # no MPI and no such header, so it gets the value of TOL_TIME.
        if self.couple_to_1d:
            tol_time_cpp = 'COUPLING_TOL_TIME'
        else:
            tol_time_cpp = f'{TOL_TIME}'

        g = Graph().parse(self.annotated_model_file_path)

        # find all delayed variables
        variables_to_delay_info = []
        for vtd in self.g.subjects(DCTERMS.type, self.variable_to_delay_uri):
            # we only expect one delay variable for each variable to delay
            dv = self.g.value(vtd, self.delay_variable_uri)
            d_amount = self.g.value(vtd, self.delay_amount_uri)
            variables_to_delay_info.append([str(vtd), str(dv), str(d_amount)])

        # print('variables_to_delay_info')
        # print(variables_to_delay_info)
        # print('\n')
            
        # # find all timecourse readouts
        # readout_variables = []
        # for d in g.subjects(DCTERMS.type, timecourse_readout_uri):
        #     readout_variables.append(str(d))
            
        # print(readout_variables)

        # find input and output variables
        port_input_variable_info = []
        input_variable_info = []
        output_variable_info = []
        for d in self.g.subjects(DCTERMS.type, self.input_variable_uri):
            # TODO most of the below for-loops could be done in this one to simplify this script
            port_output_variable_readout = self.g.value(d, self.output_variable_uri)
            external_variable_readout = self.g.value(d, self.external_variable_uri)
            control_variable_readout = self.g.value(d, self.control_variable_uri)
            info_list = [str(d)]
            if port_output_variable_readout is not None:
                info_list.append(str(port_output_variable_readout))
                if external_variable_readout is not None:
                    info_list.append(str(external_variable_readout))
                    if control_variable_readout is not None:
                        info_list.append(str(control_variable_readout))
                port_input_variable_info.append(info_list)
            else:
                input_variable_info.append(info_list)

        for d in self.g.subjects(DCTERMS.type, self.output_variable_uri):
            output_variable_info.append(str(d))

        volume_sum_variable_info = []
        for d in self.g.subjects(DCTERMS.type, self.volume_sum_variable_uri):
            volume_sum_variable_info.append(str(d))
            
        print('port_input_variable_info')
        print(port_input_variable_info)
        print('input_variable_info')
        print(input_variable_info)
        print('output_variable_info')
        print(output_variable_info)
        print('volume_sum_variable_info')
        print(volume_sum_variable_info)
        print('\n')


        # We're going to use the "model" file from the first variable to delay and only continue if all annotations use the same model...
        delayed_ids = []
        for vtd, dv, d_amount in variables_to_delay_info:
            vtd_url = urlparse(vtd)
            if vtd_url.path != self.generated_wID_model_file_path:
                print(f"generate_cpp :: ERROR: found an unexpected model file for the variable to delay {vtd}")
                exit(1)
            dv_url = urlparse(dv)
            if dv_url.path != self.generated_wID_model_file_path:
                print(f"generate_cpp :: ERROR: found an unexpected model file for the delay variable {dv}")
                exit(1)
            d_amount_url = urlparse(d_amount)
            if d_amount_url.path != self.generated_wID_model_file_path:
                print(f"generate_cpp :: ERROR: found an unexpected model file for the delay amount {d_amount}")
                exit(1)
            delayed_ids.append([vtd_url.fragment, dv_url.fragment, d_amount_url.fragment])
            
        # readout_ids = []
        # for v in readout_variables:
        #     url = urlparse(v)
        #     if url.path != model_file:
        #         print("found an unexpected model file for readout variable?!")
        #         exit
        #     readout_ids.append(url.fragment)

        port_input_variable_ids = []
        for entry in port_input_variable_info:
            if len(entry) == 2:
                port_input_var, port_output_var = entry
                external_var = None
                control_var = None
            elif len(entry) == 3:
                port_input_var, port_output_var, external_var = entry
                control_var = None
            elif len(entry) == 4:
                port_input_var, port_output_var, external_var, control_var = entry
            
            port_input_var_url = urlparse(port_input_var)
            if port_input_var_url.path != self.generated_wID_model_file_path:
                print(f"generate_cpp :: ERROR: found an unexpected model file for the port input variable {port_input_var}")
                exit(1)
            port_output_var_url = urlparse(port_output_var)
            if port_output_var_url.path != self.generated_wID_model_file_path:
                print(f"generate_cpp :: ERROR: found an unexpected model file for the port output variable {port_output_var}")
                exit(1)
            if external_var is not None:
                external_var_url = urlparse(external_var)
                if external_var_url.path != self.cpp_generated_models_dir:
                    print(f"generate_cpp :: ERROR: found an unexpected model folder for the external variable {external_var}")
                    exit(1)
                if control_var is not None:
                    control_var_url = urlparse(control_var)
                    if control_var_url.path != self.generated_wID_model_file_path:
                        print(f"generate_cpp :: ERROR: found an unexpected model file for the control variable {control_var}")
                        exit(1)
                    port_input_variable_ids.append([port_input_var_url.fragment, port_output_var_url.fragment, 
                                                    external_var_url.fragment, control_var_url.fragment])
                else:
                    port_input_variable_ids.append([port_input_var_url.fragment, port_output_var_url.fragment, 
                                                    external_var_url.fragment])
            else:
                port_input_variable_ids.append([port_input_var_url.fragment, port_output_var_url.fragment])


        input_variable_ids = []
        for input_var in input_variable_info:
            input_var_url = urlparse(input_var)
            if input_var_url.path != self.generated_wID_model_file_path:
                print(f"generate_cpp :: ERROR: found an unexpected model file for the input variable {input_var}")
                exit(1)
            input_variable_ids.append(input_var_url.fragment)

        output_variable_ids = []
        for output_var in output_variable_info:
            output_var_url = urlparse(output_var)
            if output_var_url.path != self.generated_wID_model_file_path:
                print(f"generate_cpp :: ERROR: found an unexpected model file for the output variable {output_var}")
                exit(1)
            output_variable_ids.append(output_var_url.fragment)

        volume_sum_variable_ids = []
        for volume_sum_var in volume_sum_variable_info:
            volume_sum_var_url = urlparse(volume_sum_var)
            # if volume_sum_var_url.path != self.generated_wID_model_file_path:
            if volume_sum_var_url.path != self.cpp_generated_models_dir:
                print(f"generate_cpp :: ERROR: found an unexpected model folder for the volume sum variable {volume_sum_var}")
                exit(1)
            vv = volume_sum_var_url.fragment
            v = vv.split("#")
            var_type = v[0]
            var_idx = v[1]
            var_id = v[2]
            volume_sum_variable_ids.append([var_id, var_type+"#"+var_idx])
            # volume_sum_variable_ids.append(volume_sum_var_url.fragment)

        print('port_input_variable_ids')
        print(port_input_variable_ids)
        print('volume_sum_variable_ids')
        print(volume_sum_variable_ids)
        print('\n')


        # Now we have the model file and the IDs for the variables in that model that we want to do stuff with. 
        # So we can parse the model and see if we can find the variables.
        # on windows getting a leading '/' in the filename which libCellML doesn't like...
        self.generated_wID_model_file_path = self.generated_wID_model_file_path[0:]

        # parse the model in non-strict mode to allow non CellML 2.0 models
        model = cellml.parse_model(self.generated_wID_model_file_path, False)

        # and make an annotator for this model
        # TODO this could be done before all of the loops when I turn it into one for loop.
        annotator = Annotator()
        annotator.setModel(model)

        #XXX map our IDs to the actual variables
        annotated_variables = []
        
        for vtd_id, dv_id, d_amount_id in delayed_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            vtd = annotator.variable(vtd_id)
            if vtd == None:
                print('generate_cpp :: ERROR: Unable to find a variable to delay with the id {} in the given model...'.format(vtd_id))
                exit(1)
            dv = annotator.variable(dv_id)
            if dv == None:
                print('generate_cpp :: ERROR: Unable to find a delay variable with the id {} in the given model...'.format(dv_id))
                exit(1)
            d_amount = annotator.variable(d_amount_id)
            if d_amount == None:
                print('generate_cpp :: ERROR: Unable to find a delay amount with the id {} in the given model...'.format(d_amount_id))
                exit(1)
            annotated_variables.append([[vtd, dv, d_amount], self.delay_variable_uri])
            print("DELAY annotated variable:", annotated_variables[-1])
            
        # for i in readout_ids:
        #     # get the variable (will fail if id doesn't correspond to a variable in the model)
        #     v = annotator.variable(i)
        #     if v == None:
        #         print('Unable to find a readout variable with the id {} in the given model...'.format(i))
        #         exit
        #     annotated_variables.append([v, timecourse_readout_uri])

        for entry in port_input_variable_ids:
            if len(entry) == 2:
                port_input_var_id, port_output_var_id = entry
            elif len(entry) == 3:
                port_input_var_id, port_output_var_id, external_var_id = entry
            elif len(entry) == 4:
                port_input_var_id, port_output_var_id, external_var_id, control_var_id = entry
                control_var = annotator.variable(control_var_id)
                if control_var == None:
                    print('generate_cpp :: ERROR: Unable to find a control variable with the id {} in the given model...'.format(control_var_id))
                    exit(1)
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            port_input_var = annotator.variable(port_input_var_id)
            if port_input_var == None:
                print('generate_cpp :: ERROR: Unable to find a port input variable with the id {} in the given model...'.format(port_input_var_id))
                exit(1)
            port_output_var = annotator.variable(port_output_var_id)
            if port_output_var == None:
                print('generate_cpp :: ERROR: Unable to find a port output variable with the id {} in the given model...'.format(port_output_var_id))
                exit(1)
            # TODO some check to see that the external variable is actually a variable within the cpp model??
            if len(entry) == 2:
                annotated_variables.append([[port_input_var, port_output_var], self.input_variable_uri])
            elif len(entry) == 3:
                annotated_variables.append([[port_input_var, port_output_var, external_var_id], self.input_variable_uri])
            elif len(entry) == 4:
                annotated_variables.append([[port_input_var, port_output_var, external_var_id, control_var], self.input_variable_uri])
            print("PORT INPUT annotated variable:", annotated_variables[-1])

        for input_var_id in input_variable_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            input_var = annotator.variable(input_var_id)
            if input_var == None:
                print('generate_cpp :: ERROR: Unable to find an input variable with the id {} in the given model...'.format(input_var_id))
                exit(1)
            annotated_variables.append([[input_var], self.input_variable_uri])
            print("INPUT annotated variable:", annotated_variables[-1])

        for output_var_id in output_variable_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            output_var = annotator.variable(output_var_id)
            if output_var == None:
                print('generate_cpp :: ERROR: Unable to find an output variable with the id {} in the given model...'.format(output_var_id))
                exit(1)
            annotated_variables.append([[output_var], self.output_variable_uri])
            print("OUTPUT annotated variable:", annotated_variables[-1])

        for volume_sum_var_id in volume_sum_variable_ids:
            # get the variable (will fail if id doesn't correspond to a variable in the model)
            volume_sum_var = annotator.variable(volume_sum_var_id[0])
            if volume_sum_var == None:
                print('generate_cpp :: ERROR: Unable to find a volume sum variable with the id {} in the given model...'.format(volume_sum_var_id[0]))
                exit(1)
            annotated_variables.append([[volume_sum_var, volume_sum_var_id[1]], self.volume_sum_variable_uri])
            print("VOLUME SUM annotated variable:", annotated_variables[-1])
        
        print('\n')
        

        # TODO everything above needs to be simplified and put into a single for loop.
        #  Even the below loop could be in the same for loop.
        
        # Need to work out how to map the annotations through to the variables in the generated code....
        # Generate C code for the model.

        model_dir = os.path.dirname(self.generated_wID_model_file_path)

        # resolve imports, in non-strict mode
        importer = cellml.resolve_imports(model, model_dir, False)
        # need a flattened model for analysing
        flat_model = cellml.flatten_model(model, importer)

        model_string = cellml.print_model(flat_model)
        
        with open(os.path.join(self.generated_model_subdir, self.file_prefix + '_flat.cellml'), 'w') as f:
            f.write(model_string)
        
        # analyse the model
        a = Analyser()
        
        # set the delayed variables as external
        external_variable_info = []
        for vv, uri in annotated_variables:
            if uri == self.delay_variable_uri:
                v = vv[0]
                dv = vv[1]
                d_amount = vv[2]
                flat_variable_to_delay = flat_model.component(v.parent().name()).variable(v.name())
                flat_delay_variable = flat_model.component(dv.parent().name()).variable(dv.name())
                flat_delay_amount_variable = flat_model.component(dv.parent().name()).variable(d_amount.name())
                aev = AnalyserExternalVariable(flat_delay_variable)
                aev.addDependency(flat_variable_to_delay)
                aev.addDependency(flat_delay_amount_variable)
                #
                # TODO: really need to work out how to handle other dependencies here to make sure 
                #       all required variables are up to date...
                #
                a.addExternalVariable(aev)
                # keep track of external variable information for use in generating code
                external_variable_info.append({
                    'variable': flat_variable_to_delay,
                    'delay_variable': flat_delay_variable,
                    'delay_amount_variable': flat_delay_amount_variable,
                    'analyser_variable': aev,
                    'variable_type': 'delay'
                })
                
            elif uri == self.input_variable_uri:
                if len(vv) == 1:
                    v = vv
                    input_variable = flat_model.component(v.parent().name()).variable(v.name())
                    aev = AnalyserExternalVariable(input_variable)
                    a.addExternalVariable(aev)
                    external_variable_info.append({
                        'variable': input_variable,
                        'analyser_variable': aev,
                        'variable_type': 'input'
                    })
                elif len(vv) == 2:
                    input_var = vv[0]
                    output_var = vv[1]
                    input_variable = flat_model.component(input_var.parent().name()).variable(input_var.name())
                    output_variable = flat_model.component(output_var.parent().name()).variable(output_var.name())
                    aev = AnalyserExternalVariable(input_variable)
                    a.addExternalVariable(aev)
                    external_variable_info.append({
                        'variable': input_variable,
                        'port_variable': output_variable,
                        'analyser_variable': aev,
                        'variable_type': 'input'
                    })
                elif len(vv) == 3:
                    input_var = vv[0]
                    output_var = vv[1]
                    ext_variable = vv[2]
                    input_variable = flat_model.component(input_var.parent().name()).variable(input_var.name())
                    output_variable = flat_model.component(output_var.parent().name()).variable(output_var.name())
                    print("external variable:", ext_variable)
                    splt = ext_variable.split('#')
                    ext_variable_type = splt[0]
                    ext_variable_idx = splt[1]
                    ext_variable_inlet_or_outlet = splt[2]
                    ext_variable_flow_or_pressure_bc = splt[3]
            
                    aev = AnalyserExternalVariable(input_variable)
                    a.addExternalVariable(aev)
                    external_variable_info.append({
                        'variable': input_variable,
                        'port_variable': output_variable,
                        'ext_variable_idx': ext_variable_idx,
                        'ext_variable_type': ext_variable_type,
                        'coupled_to_type': 'FV_1d' if ext_variable_type.startswith('FV_1d') else 'unknown',
                        'inlet_or_outlet': ext_variable_inlet_or_outlet,
                        'bc_inlet0_or_outlet1': 0 if ext_variable_inlet_or_outlet == 'inlet' else 1,
                        'flow_or_pressure_bc': ext_variable_flow_or_pressure_bc,
                        'analyser_variable': aev,
                        'control_variable': None,
                        'control_variable_index': None,
                        'variable_type': 'input'
                    })
                elif len(vv) == 4:
                    # TODO simplify so i'm not rewriting all of this for each case
                    input_var = vv[0]
                    output_var = vv[1]
                    ext_variable = vv[2]
                    print("external variable:", ext_variable)
                    cont_var = vv[3]
                    input_variable = flat_model.component(input_var.parent().name()).variable(input_var.name())
                    output_variable = flat_model.component(output_var.parent().name()).variable(output_var.name())
                    cont_variable = flat_model.component(cont_var.parent().name()).variable(cont_var.name())
                    splt = ext_variable.split('#')
                    ext_variable_type = splt[0]
                    ext_variable_idx = splt[1]
                    ext_variable_inlet_or_outlet = splt[2]
                    ext_variable_flow_or_pressure_bc = splt[3]
            
                    aev = AnalyserExternalVariable(input_variable)
                    a.addExternalVariable(aev)
                    external_variable_info.append({
                        'variable': input_variable,
                        'port_variable': output_variable,
                        'ext_variable_idx': ext_variable_idx,
                        'ext_variable_type': ext_variable_type,
                        'coupled_to_type': 'FV_1d' if ext_variable_type.startswith('FV_1d') else 'unknown',
                        'inlet_or_outlet': ext_variable_inlet_or_outlet,
                        'bc_inlet0_or_outlet1': 0 if ext_variable_inlet_or_outlet == 'inlet' else 1,
                        'flow_or_pressure_bc': ext_variable_flow_or_pressure_bc,
                        'analyser_variable': aev,
                        'control_variable': cont_variable,
                        'control_variable_index': None,
                        'variable_type': 'input'
                    })
                print("input variable:", external_variable_info[-1], len(vv))
                print("****")
                
            elif uri == self.output_variable_uri:
                # v = vv
                v = vv[0]
                output_variable = flat_model.component(v.parent().name()).variable(v.name())
                external_variable_info.append({
                    'variable': output_variable,
                    'variable_type': 'output'
                })
                # TODO I need to include specification of the external variable here? i.e. the name 
                # or index of the variable from the Cpp code?

            elif uri == self.volume_sum_variable_uri:
                v = vv[0]
                volume_sum_var = flat_model.component(v.parent().name()).variable(v.name())
                
                volume_sum_type = vv[1].split("#")
                v_type = volume_sum_type[0]
                v_idx = volume_sum_type[1]

                aev = AnalyserExternalVariable(volume_sum_var)
                a.addExternalVariable(aev)
                external_variable_info.append({
                    'variable': volume_sum_var,
                    'ext_variable_idx': v_idx,
                    'coupled_to_type': 'FV_1d' if v_type.startswith('FV_1d') else 'unknown',
                    'analyser_variable': aev,
                    'variable_type': 'volume_sum'
                })
                print("volume sum variable:", external_variable_info[-1], len(vv))
                print("****")

        
        a.analyseModel(flat_model)
        analysed_model = a.model()

        if self.DEBUG:
            # parse_model seems to print most of the necessary issues, so we don't need to print them here
            # unless debugging... To check with Hugh
            libcellml_utils.print_issues(a)


        print(f"analysed model has type {analysed_model.type()} . Is it ODE type? {analysed_model.type()==AnalyserModel.Type.ODE}")
        # print(f"analysed model has type {analysed_model.type()} . Is it DAE type? {analysed_model.type()==AnalyserModel.Type.DAE}")
        if analysed_model.type() != AnalyserModel.Type.ODE:
            # e.g. a DAE or NLA model (with algebraic loops), or an invalid one: libCellML's issues say why
            if not self.DEBUG:
                libcellml_utils.print_issues(a)
            print(f"generate_cpp :: ERROR: the model is not a valid ODE model (libCellML analysis: "
                  f"{AnalyserModel.typeAsString(analysed_model.type())}), see the issues above. The Cpp solvers "
                  f"support ODE models only. Aborting...")
            exit(1)
        # if not analysed_model.isValid():
        #     print("generate_cpp :: ERROR: model is not valid, aborting...")
        #     exit()

        # get the information for the variables to delay
        for ext_variable in external_variable_info:
            # print("external variable:", ext_variable)
            ev = ext_variable['variable']
            ev_type = ext_variable['variable_type']
            bc_inlet_or_outlet = ext_variable.get('bc_inlet0_or_outlet1') # 1d-0d coupling variables only (not delays)

            idxev = -1  
            epv = -1
            idx1d0d = -1
            if 'coupled_to_type' in ext_variable:
                if ext_variable['coupled_to_type']=='FV_1d':
                    idxev = ext_variable['ext_variable_idx'] 
                    if ev_type == 'volume_sum':
                        for j in range(len(self.conn_1d_0d_info)):
                            if "port_volume_sum" in self.conn_1d_0d_info[str(j+1)]:
                                if (self.conn_1d_0d_info[str(j+1)]["port_volume_sum"]==1 
                                    and str(self.conn_1d_0d_info[str(j+1)]["vess0d_idx"])==idxev):
                                    idx1d0d = j+1
                                    break
                        if idx1d0d<0:
                            print(f"generate_cpp :: ERROR: 1d-0d connection not found in self.conn_1d_0d_info for volume_sum variable coupled to FV_1d type.")
                            exit(1)
                    else:
                        epv = ext_variable['port_variable']
                        for j in range(len(self.conn_1d_0d_info)):
                            if (str(self.conn_1d_0d_info[str(j+1)]["vess1d_idx"])==idxev
                                and self.conn_1d_0d_info[str(j+1)]["cellml_bc_in0_or_out1"]==bc_inlet_or_outlet):
                                idx1d0d = j+1
                                break
                        if idx1d0d<0:
                            print(f"generate_cpp :: ERROR: 1d-0d vessel connection not found in self.conn_1d_0d_info for ext_variable_idx {idxev} coupled to FV_1d type.")
                            exit(1)

            # print("idx0d1d:", idx1d0d)
            
            avs = analysed_model.variables()
            for av in avs:
                v = av.variable()
                
                # if analysed_model.areEquivalentVariables(v, ext_variable['variable']):
                if analysed_model.areEquivalentVariables(v, ev):
                    ext_variable['variable_index'] = av.index()
                    ext_variable['state_or_variable'] = 'variable'
                    if idx1d0d>0:
                        self.conn_1d_0d_info[str(idx1d0d)]["cellml_idx"] = av.index() # in the meantime, filling also self.conn_1d_0d_info
                    print(v.name(), ev.name(), " || variable index", ext_variable['variable_index'])

                if 'port_variable' in ext_variable.keys():
                    # if analysed_model.areEquivalentVariables(v, ext_variable['port_variable']):
                    if analysed_model.areEquivalentVariables(v, epv):
                        ext_variable['port_variable_index'] = av.index()
                        if idx1d0d>0:
                            self.conn_1d_0d_info[str(idx1d0d)]["port_idx"] = av.index() # in the meantime, filling also self.conn_1d_0d_info
                            self.conn_1d_0d_info[str(idx1d0d)]["port_state0_or_var1"] = 1 # variable
                        print(v.name(), epv.name(), " || port variable index", ext_variable['port_variable_index'])

                if 'control_variable' in ext_variable.keys() and ext_variable['control_variable'] != None:
                    ecv = ext_variable['control_variable']
                    # if analysed_model.areEquivalentVariables(v, ext_variable['control_variable']):
                    if analysed_model.areEquivalentVariables(v, ecv):
                        ext_variable['control_variable_index'] = av.index()
                        if idx1d0d>0:
                            self.conn_1d_0d_info[str(idx1d0d)]["R_T_variable_idx"] = av.index() # in the meantime, filling also self.conn_1d_0d_info
                        print(v.name(), ecv.name(), " || control variable index", ext_variable['control_variable_index'])

                if ext_variable['variable_type'] == 'delay':
                    if analysed_model.areEquivalentVariables(v, ext_variable['delay_variable']):
                        ext_variable['delay_variable_index'] = av.index()
                    if analysed_model.areEquivalentVariables(v, ext_variable['delay_amount_variable']):
                        ext_variable['delay_amount_index'] = av.index()
                        ext_variable['delay_amount_type'] = av.type()
                        ext_variable['delay_amount_value'] = av.variable().initialValue()
            
            astates = analysed_model.states()
            for astate in astates:
                state = astate.variable()
                
                # if state.name() == ext_variable['variable'].name(): 
                # if state.name() == ev.name(): #XXX Bea: I think this was wrong, replaced by following line
                if analysed_model.areEquivalentVariables(state, ev): 
                    ext_variable['state_index'] = astate.index()
                    ext_variable['state_or_variable'] = 'state'
                    if idx1d0d>0:
                        self.conn_1d_0d_info[str(idx1d0d)]["cellml_idx"] = astate.index() # in the meantime, filling also self.conn_1d_0d_info
                    print(state.name(), ev.name(), " || state index", ext_variable['state_index'])
                
                if 'port_variable' in ext_variable.keys():
                    # if state.name() == ext_variable['port_variable'].name(): 
                    # if state.name() == epv.name(): #XXX Bea: I think this was wrong, replaced by following line
                    if analysed_model.areEquivalentVariables(state, epv): 
                        ext_variable['port_state_index'] = astate.index()
                        if idx1d0d>0:
                            self.conn_1d_0d_info[str(idx1d0d)]["port_idx"] = astate.index() # in the meantime, filling also self.conn_1d_0d_info
                            self.conn_1d_0d_info[str(idx1d0d)]["port_state0_or_var1"] = 0 # state
                        print(state.name(), epv.name(), " || port state index", ext_variable['port_state_index'])
            print("****")

        # Delay variables: delayed variable = variable to delay at (t - delay amount). Each one is computed from the
        # history of the variable to delay (values at the end of the time steps, interpolated linearly), before the
        # other variables (see getDelayVariables(), initDelayHistories() and storeDelayHistories() in the generated code).
        delay_variables = []
        for ext_variable in external_variable_info:
            if ext_variable['variable_type'] != 'delay':
                continue
            dv = ext_variable['delay_variable']
            name = f"{dv.parent().name()}/{dv.name()} = {ext_variable['variable'].name()}(t - {ext_variable['delay_amount_variable'].name()})"
            if ext_variable.get('state_or_variable') == 'state':
                source = f"states[{ext_variable['state_index']}]"
            elif ext_variable.get('state_or_variable') == 'variable':
                source = f"variables[{ext_variable['variable_index']}]"
            else:
                print(f"generate_cpp :: ERROR: delay variable {name}: the variable to delay is not a state or an algebraic "
                      "variable of the analysed model")
                exit(1)
            if 'delay_variable_index' not in ext_variable or 'delay_amount_index' not in ext_variable:
                print(f"generate_cpp :: ERROR: delay variable {name}: the delayed variable or the delay amount is not in the "
                      "analysed model")
                exit(1)
            if ext_variable['delay_amount_type'] not in (AnalyserVariable.Type.CONSTANT, AnalyserVariable.Type.COMPUTED_CONSTANT):
                print(f"generate_cpp :: ERROR: delay variable {name}: the delay amount must be a constant (or computed from "
                      "constants only): a delay that changes in time is not implemented")
                exit(1)
            delay_variables.append({'name': name, 'source': source, 'delayed_index': ext_variable['delay_variable_index'],
                                    'amount_index': ext_variable['delay_amount_index']})
            try:
                delay_value = float(ext_variable.get('delay_amount_value') or 'nan')
            except ValueError:
                delay_value = float('nan')
            if delay_value <= 0.0:
                print(f"generate_cpp :: ERROR: delay variable {name}: the delay amount must be > 0 ({delay_value} given)")
                exit(1)
            if delay_value < float(self.dtSolver):
                print(f"generate_cpp :: WARNING: delay variable {name}: the delay amount {delay_value} is shorter than the "
                      f"solver time step {self.dtSolver}: the delayed value then lags by up to one time step")
        if delay_variables:
            print(f"generate_cpp :: NOTE: this model has {len(delay_variables)} delay variable(s):\n  "
                  + "\n  ".join(d['name'] for d in delay_variables)
                  + "\n  Their implementation in the cpp solvers is basic and might need improvements (check the results): "
                  "each delayed value is interpolated linearly in the history of the values at the end of the time steps, "
                  "and takes the initial value of the variable to delay for t < delay.")

        # Variables exchanged with the 1d solver, used to split the computation of the variables around the exchange
        # (see split_variables_at_coupling_exchange() and analyse_coupling_connections()):
        # - coupling_received: received from the 1d solver at each evaluation of the rates (getExternalVariables());
        # - coupling_volume_sum: 1d volumes, received only at the end of each time step (getExternalVariablesVolume());
        # - coupling_connections: the one-to-one 1d-0d connections, in the order of the exchange.
        coupling_received = set()
        coupling_volume_sum = set()
        coupling_connections = []
        variable_names = {}
        for av in analysed_model.variables():
            try:
                variable_names[av.index()] = f"{av.variable().parent().name()}/{av.variable().name()}"
            except Exception:
                variable_names[av.index()] = av.variable().name()
        if self.couple_to_1d and self.conn_1d_0d_info is not None:
            for ext_variable in external_variable_info:
                if ext_variable.get('control_variable') is not None and ext_variable.get('control_variable_index') is None:
                    raise RuntimeError(f"1d-0d coupling: the resistance {ext_variable['control_variable'].name()} sent "
                                       f"with the coupling variable {ext_variable['variable'].name()} is not an algebraic "
                                       "variable or a constant of the analysed model (a state cannot be sent as resistance).")
            # global connections (e.g. volume sum), after the one-to-one ones (also checks their types and order)
            global_keys = {key for key, _ in global_coupling_connections(self.conn_1d_0d_info)}
            for i in range(len(self.conn_1d_0d_info)):
                key = str(i+1)
                conn = self.conn_1d_0d_info[key]
                if key in global_keys:
                    if conn["cellml_idx"] >= 0:
                        coupling_volume_sum.add(conn["cellml_idx"])
                    continue
                if len(coupling_connections) != i:
                    raise RuntimeError("1d-0d coupling: the one-to-one connections must come before the volume sum "
                                       "connections in the connection information (as written by ModelParsers).")
                where = (f"1d-0d connection {key} (1d vessel {conn.get('vess1d_name', conn['vess1d_idx'])}, "
                         f"{'inlet' if conn['cellml_bc_in0_or_out1'] == 0 else 'outlet'} of the 0d module)")
                if conn["cellml_idx"] < 0 or conn["port_idx"] < 0:
                    raise RuntimeError(f"{where}: received variable (index {conn['cellml_idx']}) or sent variable "
                                       f"(index {conn['port_idx']}) not found in the analysed model.")
                coupling_received.add(conn["cellml_idx"])
                coupling_connections.append({
                    'key': key, 'where': where, 'received': conn["cellml_idx"], 'sent': conn["port_idx"],
                    'sent_is_state': conn["port_state0_or_var1"] == 0,
                    'flow_bc': conn["cellml_bc_flow0_or_press1"] == 0,
                    'sign': 1 if conn["cellml_bc_in0_or_out1"] == 0 else -1, # 1d outlet (0d inlet): +1; 1d inlet: -1
                    'resistance': conn["R_T_variable_idx"] if conn["R_T_variable_idx"] >= 0 else None})

        # generate code from the analysed model
        gen = Generator()
        # using the C profile to generate C code
        # TODO BIG CHANGE
        # TODO I should create my own profile with most of the below code so I don't have to change it when changes are made to libcellml
        profile = GeneratorProfile(GeneratorProfile.Profile.C)
        profile.setInterfaceFileNameString(f'{self.output_cpp_file_name}.h')
        # MathML xor: the C profile calls its helper function "xor", which is a reserved word in C++ (alternative token
        # of the operator ^), so the generated code would not compile. Same function, with another name.
        profile.setXorString('xorFunc')
        profile.setXorFunctionString('double xorFunc(double x, double y)\n'
                                     '{\n'
                                     '    return (x != 0.0) ^ (y != 0.0);\n'
                                     '}\n')
        # MathML not: the C profile writes the operator ! without parentheses around its argument, e.g. not(t < 0.5)
        # becomes "!voi < 0.5", which C++ reads as "(!voi) < 0.5" (true for any t > 0): a function call instead, with
        # its argument in parentheses ("not" is also a reserved word in C++)
        profile.setHasNotOperator(False)
        profile.setNotString('notFunc')
        profile.setNotFunctionString('double notFunc(double x)\n'
                                     '{\n'
                                     '    return x == 0.0;\n'
                                     '}\n')
        gen.setProfile(profile)
        gen.setModel(analysed_model)

        # Variables computed at each evaluation of the rates (libCellML's computeRates() and computeVariables()):
        # those of computeRates() first, in their order, then those only in computeVariables(). libCellML's
        # computeVariables() does not recompute the variables that depend only on voi and constants (e.g. a prescribed
        # inlet flow), because it expects computeRates() to have been called before at the same voi: listing its lines
        # first would compute the variables that use them with their values from the previous evaluation.
        # They are then split around the exchange with the 1d solver, and each 1d-0d connection gets its way of
        # exchanging its values (see analyse_coupling_connections()).
        rates_var_lines, variables_var_lines = libcellml_variable_lines(gen.implementationCode().split('\n'))
        non_external_lines = []
        for line2 in rates_var_lines + variables_var_lines:
            if 'external' not in line2 and line2 not in non_external_lines:
                non_external_lines.append(line2)
        # no variable computed from a derivative (rates[k]): they are computed before the rates
        state_names = {}
        for astate in analysed_model.states():
            try:
                state_names[astate.index()] = f"{astate.variable().parent().name()}/{astate.variable().name()}"
            except Exception:
                state_names[astate.index()] = astate.variable().name()
        check_variables_use_no_rates(non_external_lines, variable_names, state_names)
        check_variables_computation_order(non_external_lines, rates_var_lines + variables_var_lines,
                                        coupling_received | coupling_volume_sum | {d['delayed_index'] for d in delay_variables})
                                        # a single set built from three unions with '|'
        lines_before, lines_after, depends_on = split_variables_at_coupling_exchange(non_external_lines, coupling_received)
        if lines_after:
            print(f"1d-0d coupling: {len(lines_after)} variables depend on the variables received from the 1d solver, "
                  "and are computed after the exchange with it:\n  " + "\n  ".join(line2.strip() for line2 in lines_after))
        coupling_send_code, coupling_plans = analyse_coupling_connections(coupling_connections, lines_after, depends_on,
                                                                          variable_names)
        for conn, plan in zip(coupling_connections, coupling_plans):
            if plan['affine']:
                print(f"{conn['where']}: {plan['description']}")
            if plan['converted']:
                # the 1d solver exchanges it as a flow bc: it receives P0D and R, and sends back the flow
                self.conn_1d_0d_info[conn['key']]["cellml_bc_flow0_or_press1"] = 0
                self.conn_1d_0d_info[conn['key']]["port_flow0_or_press1"] = 1
                self.conn_1d_0d_info[conn['key']]["pressure_bc_as_flow_bc"] = 1

        #XXX saving final 1d-0d connectvity information in generated model directory
        if self.conn_1d_0d_info is not None and self.couple_to_1d:
            # named after the whole file_prefix, as the coupler reads <networkName>_coupler1d0d.json with
            # networkName = file_prefix (coupler_config.json), and the 1d solver its own copy with the same name
            json_filename = self.file_prefix+"_coupler1d0d.json"
            
            # vess1d_name is used only by the generators (the coupler reads this file as integer values only)
            conn_1d_0d_info_json = {k: {kk: vv for kk, vv in conn.items() if kk != "vess1d_name"}
                                    for k, conn in self.conn_1d_0d_info.items()}
            with open(self.generated_model_subdir+"/"+json_filename, "w") as f:
                json.dump(conn_1d_0d_info_json, f, indent=4)
            with open(self.cpp_generated_models_dir+"/"+json_filename, "w") as f:
                json.dump(conn_1d_0d_info_json, f, indent=4)


        #XXX HEADER FILE

        preHeaderStuff = f"""
#include <stdlib.h>
#include <memory>
#include <map>
#include <string>
#include <sstream>
#include <functional>
#include <cstring>  
#include <fstream>
#include <vector>
"""
        for external_header in self.external_headers:
            preHeaderStuff += f'#include "{external_header}"\n'

        if self.solver == 'CVODE':
            preHeaderStuff += """
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h> 
#include <sundials/sundials_types.h>

        """
        elif self.solver == 'PETSC':
            preHeaderStuff += """
#include <petscsys.h>
#include <petscts.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>

        """

        interFaceCodePreClass = ''
        interFaceCodeInClass = ''
        pre_class = True
        for line in gen.interfaceCode().split('\n'):
            # print(line)
            if 'extern ' in line:
                if 'VERSION' in line:
                    line = line.replace('VERSION', 'VERSION_')
                if 'VERSION' not in line:
                    line = line.replace('extern ', '')
            if '(* ExternalVariable)' in line:
                continue
            if ', ExternalVariable externalVariable' in line:
                line = line.replace(', ExternalVariable externalVariable', '')
            if 'const VariableInfo' in line:
                line = 'static ' + line
            if line.startswith('const size_t STATE_COUNT'):
                pre_class = False

            if ('initialiseVariables(' in line or 'computeComputedConstants(' in line or 'computeRates(' in line or 'computeVariables(' in line):
                # print(line)
                if ('computeRates(' in line or 'computeVariables(' in line):
                    if 'double voi' in line:
                        line = line.replace('double voi', 'double voi, double dt')
                if 'initialiseVariables(' in line:
                    if 'double voi' not in line:
                        line = line.replace('(', '(double voi, ')
                if 'double voi' in line:
                    line = line.replace('double voi', 'double voiLoc')
                if 'dt' in line:
                    line = line.replace('dt', 'dtLoc')
                if 'states' in line:
                    line = line.replace('states', 'statesLoc')
                if 'rates' in line:
                    line = line.replace('rates', 'ratesLoc')
                if 'variables' in line:
                    line = line.replace('variables', 'varLoc')
                # print(line)

            if pre_class:
                interFaceCodePreClass += line + '\n'
            else:
                interFaceCodeInClass += '    ' + line + '\n'
            # print(line)


        classInitHeader = ""

        if self.solver == 'CVODE':
            classInitHeader += """
// forward declaration for userOdeData class
class UserOdeData;
        """
        elif self.solver == 'PETSC':
            classInitHeader += """
// forward declarations for PETSc objects
struct _p_TS;
typedef struct _p_TS* TS; // abstract PETSc object that manages integrating an ODE.
struct _p_Vec;
typedef struct _p_Vec* Vec; // abstract PETSc vector object.
struct _p_Mat;
typedef struct _p_Mat* Mat; // abstract PETSc matrix object used to manage all linear operators in PETSc,
                            // even those without an explicit sparse representation (such as matrix-free operators).
struct _p_SNES;
typedef struct _p_SNES* SNES; // abstract PETSc object that manages the nonlinear solves in PETSc
struct _p_KSP;
typedef struct _p_KSP* KSP; // abstract PETSc object that manages the linear solves in PETSc 
                            // (even those such as direct factorization-based solvers that do not use Krylov accelerators).
struct _p_PC;
typedef struct _p_PC* PC; // abstract PETSc object that manages all preconditioners including direct solvers such as PCLU.

// forward declaration for userOdeData class
class UserOdeData;
        """

        classInitHeader += """
// this is the the 0D model class definition
// it contains the model variables and the functions to compute the rates and variables
class Model0d {
public:
    // constructor
    Model0d();
    // destructor
    ~Model0d();
        """

        otherHeaderInits = f"""
    void computeNonExternalVariables(double voiLoc, double *statesLoc, double *ratesLoc, double *varLoc);
    void computeCouplingDependentVariables(double voiLoc, double *statesLoc, double *ratesLoc, double *varLoc);
    void computeCouplingSentValues(double voiLoc, double *statesLoc, double *ratesLoc, double *varLoc);
    // values sent to the 1d solver for each 1d-0d connection i: couplingSend[2*i] = pressure P0D (or flow),
    // couplingSend[2*i+1] = resistance R; the 1d solver imposes P0D + R*q (1d outlet) or P0D - R*q (1d inlet)
    double couplingSend[{max(2*len(coupling_connections), 2)}];
    void getDelayVariables(double voiLoc, double dtLoc, double *statesLoc, double *varLoc);
    void initDelayHistories();  // delay variables: history of the variables to delay, from their initial values
    void storeDelayHistories(); // delay variables: values at the end of each time step
    void getExternalVariables(double voiLoc, double dtLoc, double *statesLoc, double *varLoc);
    void getExternalVariablesVolume(double *varLoc);

    void initialiseVariablesAndComputeConstants();

    const char* initStatesPath;
    const char* initVarsPath;
    void loadFromFile(const std::string& filename, double* targetArray, const Model0d::VariableInfo infoArray[], size_t infoSize, bool useFirstRow = false);
    void initialise(double voiLoc, double *statesLoc, double *ratesLoc, double *varLoc, const char* statesPath, const char* varsPath, bool useFirstRow = false);

    std::string solver_0d;
    void set_ode_solver(std::string ODEsolver);

    std::ofstream outFileStates;
    std::ofstream outFileVars;
    std::vector<int> idx_states_to_output;
    std::vector<int> idx_vars_to_output;
    void openOutputFiles(std::string outDir);
    void writeOutput(double voiLoc);
    void closeOutputFiles();
        """

    
        otherHeaderInits += """
    void solveOneStep(double dtLoc);
    double roundTime(double t) const;
        """

        otherHeaderInits += """
    double voi;
    double voiEnd;
    double dt;
    double tolTime;
    double * states;
    double * rates;
    double * variables;
        """

        delayHistoriesMember = """
    std::vector<DelayHistory> delayHistories; // one per delay variable, see getDelayVariables()
        """

        if self.couple_to_1d:
            otherHeaderInits += """
    int N1d0d;
    int N1d0dTot;
    int N1d0dGlobal; // global connections (e.g. volume sum), after the one-to-one ones: rows of the TAG_VOL messages
            """

        if self.solver == 'CVODE':
            otherHeaderInits += """
    double * states0; 
    SUNContext context;
    void * solver;
    N_Vector y;
    SUNMatrix matrix;
    SUNLinearSolver linearSolver;

    // UserOdeData* userData= nullptr;
    UserOdeData* userData;
        """
        elif self.solver == 'PETSC':
            otherHeaderInits += """
    // PETSc objects
    TS ts;
    Vec y;
    Vec rates_vec; // optional
    Mat J;
    SNES snes;
    KSP ksp;
    PC pc;

    UserOdeData* userData;
        """
        else:
            otherHeaderInits += """
    double * states0; 
    double wRK4[4];
    double * k1;
    double * k2;
    double * k3;
    double * k4;
            """


        otherHeaderInits += """
    using FunctionType = std::function<void(double, double, double*, double*, double*)>;
        """

        # using computeRatesType = void (*)(double, double *, double *, double *);
        # static int func(double voi, N_Vector y, N_Vector ydot, void *userData);

        if self.couple_to_1d:
            otherHeaderInits += """
    // MPI coupling with the 1d solver, through the coupler (see coupler_mpi_common.h)
    int coupler_rank; // world rank of the coupler
    int initCoupling(int couplerRank, double T0, double pre_time, double sim_time);
    std::vector<std::map<std::string, int>> compiledConnections() const; // 1d-0d connections at generation
    void finaliseCoupling();

    int DATA_LENGTH;
    std::vector<double> zero_data;       // N1d0d x DATA_LENGTH (row-major), 0d -> coupler (coupling values for all the 1D-0D connections stored in one flat vector)
    std::vector<double> parent_data;     // N1d0d x DATA_LENGTH (row-major), coupler -> 0d (coupling values for all the 1D-0D connections stored in one flat vector)
    // This flat layout lets all connections go out/come in in one MPI message.
    std::vector<double> zero_data_dt;    // {time, time step}, 0d -> coupler
    std::vector<double> parent_data_dt;  // {time, time step}, coupler -> 0d
    std::vector<double> parent_data_vol; // N1d0dGlobal x DATA_LENGTH (row-major), coupler -> 0d: global connections (TAG_VOL)
            """

        classFinisherHeader = """
};
        """


        #XXX SOURCE FILE

        preSourceStuff = f"""
#include <stddef.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <cstring>  
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
        """

        if self.couple_to_1d: #XXX
            preSourceStuff += """
#include "coupler_mpi_common.h"
        """

        if self.solver == 'CVODE':
            preSourceStuff += """
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h> 
#include <sundials/sundials_types.h>
        """
        elif self.solver == 'PETSC':
            preSourceStuff += """
#include <petscsys.h>
#include <petscts.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>
#include <limits>

// Check a PETSc call outside the PETSc callbacks: if it fails, PETSc prints its error message, then this stops the
// program (in a coupled run, mpirun then stops the coupler and the 1d solver). PetscCallVoid() would only return from
// the calling function, and the time loop would go on (e.g. repeating forever a time step that cannot be taken).
#define PETSC_CHECK(call)                                                                                  \\
    do {                                                                                                   \\
        const PetscErrorCode ierrCheck = (call);                                                           \\
        if (ierrCheck != PETSC_SUCCESS) {                                                                  \\
            std::cerr << "0d solver :: ERROR: PETSc call " << #call << " failed with error code "          \\
                      << static_cast<int>(ierrCheck) << " (" << __FILE__ << ":" << __LINE__ << "). Exiting." \\
                      << std::endl;                                                                        \\
            exit(1);                                                                                       \\
        }                                                                                                  \\
    } while (0)
        """
            # PETSc time integration scheme, set at generation (see get_petsc_solver()). In a coupled model, the
            # coupler checks at start-up that it is the scheme of its configuration file and of the 1d solver.
            preSourceStuff += "\n// PETSc time integration scheme, set at generation (petsc_solver in solver_info of the user inputs)\n"
            for scheme, description in PETSC_SOLVERS.items():
                preSourceStuff += f"{'' if scheme == self.petsc_solver else '// '}#define SOLVER_{scheme} // {description}\n"
            preSourceStuff += "\n"

        with open(generators_dir_path+'/cppGeneratorTemplateFunctions.cpp', 'r') as file:
            lines_tmp = file.readlines()
        
        iStart = None
        iEnd = None
        for iL, line in enumerate(lines_tmp):
            if "Functions for index-name mapping" in line:
                iStart = iL
                break
        for iL, line in enumerate(lines_tmp[iStart+1:], start=iStart+1):
            if "End functions" in line:
                iEnd = iL
                break
        for iL, line in enumerate(lines_tmp[iStart:iEnd+1], start=iStart):
            preSourceStuff += line #+ '\n'
        preSourceStuff += '\n'

        # split implementation code in two so we can change it into a class
        preClassStuff = ''
        
        classInit = """
Model0d::Model0d() :
        """
        postClassInit = ''

        # Members initialised by the constructor, in the order of their declaration in the class (otherwise -Wreorder):
        # they follow STATE_COUNT and VARIABLE_COUNT, which come from libCellML's code below.
        memberInits = f"""
    initStatesPath(nullptr),
    initVarsPath(nullptr),
    solver_0d(""),
    voi(0.0),
    voiEnd(0.0),
    dt(0.0),
    tolTime({tol_time_cpp}),
    states(nullptr),
    rates(nullptr),
    variables(nullptr),"""

        if self.couple_to_1d:
            N1d0dTot = len(self.conn_1d_0d_info)
            global_conns = global_coupling_connections(self.conn_1d_0d_info)
            N1d0dGlobal = len(global_conns)
            N1d0d = N1d0dTot - N1d0dGlobal
            memberInits += f"""
    N1d0d({N1d0d}),
    N1d0dTot({N1d0dTot}),
    N1d0dGlobal({N1d0dGlobal}),"""

        if self.solver == 'CVODE':
            memberInits += """
    states0(nullptr),
    userData(nullptr),"""
        elif self.solver == 'PETSC':
            memberInits += """
    ts(nullptr),
    y(nullptr),
    rates_vec(nullptr),
    J(nullptr),
    snes(nullptr),
    ksp(nullptr),
    pc(nullptr),
    userData(nullptr),"""
        else: # elif self.solver == 'RK4':
            memberInits += """
    states0(nullptr),
    k1(nullptr),
    k2(nullptr),
    k3(nullptr),
    k4(nullptr),"""
        memberInits = memberInits.rstrip(',') # the last one is followed by the body of the constructor
    #     else:
    #         N1d0d = 0
    #         N1d0dTot = 0
    #         classInit += f"""
    # N1d0d({N1d0d}),
    # N1d0dTot({N1d0dTot}),
    #     """
           

            # classInit += """    cellml_index_to_vessel1d_info{
            # """
            # num_vessel1d_connections = 0 
            # for idx, ext_variable in enumerate(external_variable_info):
            #     if idx != 0:
            #         classInit += ',\n'
            #     if ext_variable['coupled_to_type'] == 'FV_1d':
            #         classInit += f"""       {{ "{ext_variable["variable_index"]}", {{ 
            #             {{ "vessel1d_idx", {ext_variable["ext_variable_idx"]}}}, 
            #             {{ "bc_inlet0_or_outlet1", {ext_variable["bc_inlet0_or_outlet1"]} }}"""
            #         if 'port_variable_index' in ext_variable.keys():
            #             classInit += f""",
            #             {{ "port_variable_idx", {ext_variable["port_variable_index"]}}}"""
            #         elif 'port_state_index' in ext_variable.keys():
            #             classInit += f""",
            #             {{ "port_state_idx", {ext_variable["port_state_index"]}, }}"""
            #         if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
            #             classInit += f""",
            #             {{ "control_variable_idx", {ext_variable["control_variable_index"]}, }}"""
            #         classInit += """}}"""
            #         num_vessel1d_connections += 1
            # classInit += '},\n'
            # classInit += f'    num_vessel1d_connections({num_vessel1d_connections}),\n'

            # # now create a vessel1d_info vector of dicts for each connected variable
            # classInit += """    vessel1d_info{ """
            # for idx, ext_variable in enumerate(external_variable_info):
            #     if idx != 0:
            #         classInit += ',\n'
            #     if ext_variable['coupled_to_type'] == 'FV_1d':
            #         classInit += f"""       
            #         {{  {{ "cellml_idx", {ext_variable["variable_index"]}}}, 
            #             {{ "vessel1d_idx", {ext_variable["ext_variable_idx"]}}}, 
            #             {{ "bc_inlet0_or_outlet1", {ext_variable["bc_inlet0_or_outlet1"]} }} """
            #         if 'port_variable_index' in ext_variable.keys():
            #             classInit += f""",
            #             {{ "port_variable_idx", {ext_variable["port_variable_index"]}, }}
            #             """
            #         elif 'port_state_index' in ext_variable.keys():
            #             classInit += f""",
            #             {{ "port_state_idx", {ext_variable["port_state_index"]}, }}
            #             """
            #         if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
            #             classInit += f""",
            #             {{ "control_variable_idx", {ext_variable["control_variable_index"]}, }}
            #             """
            #         classInit += "}"
            # classInit += '    },\n'

        
        pre_class = True
        in_class_init = False

        # TODO The below is all really susceptible to changes in libcellml, I should create my own profile
        lines = gen.implementationCode().split('\n')
        max_func_found = any(l.startswith('double max') for l in lines)
        
        lines_to_pass = []
        lines_to_pass_rates = []
        lines_to_pass_variables = []
        init_lines_to_mod = []
        for iL, line in enumerate(lines):
            # print(iL, line)
            if line.startswith('const size_t STATE_COUNT'):
                pre_class = False
                in_class_init = True

            if line.startswith('const VariableInfo VOI_INFO'):
                pre_class = True
                in_class_init = False

            if 'VERSION' in line:
                line = line.replace('VERSION', 'VERSION_')
            
            # max_func_found = False
            # for line2 in gen.implementationCode().split('\n'):
            #     if line2.startswith('double max'):
            #         max_func_found = True

            if max_func_found:
                if line.startswith('double max'):
                    pre_class = False
                    in_class_init = False
            else:
                if line.startswith('double * createStatesArray'):
                    pre_class = False
                    in_class_init = False
            # if line.startswith('double * createStatesArray'):
            # # if line.startswith('double max'):
            #     in_class_init = False

            if pre_class:
                if line.startswith('const VariableInfo VOI_INFO'):
                    line = line.replace('VariableInfo VOI_INFO', 'Model0d::VariableInfo Model0d::VOI_INFO')
                if line.startswith('const VariableInfo STATE_INFO'):
                    line = line.replace('VariableInfo STATE_INFO', 'Model0d::VariableInfo Model0d::STATE_INFO')
                if line.startswith('const VariableInfo VARIABLE_INFO'):
                    line = line.replace('VariableInfo VARIABLE_INFO', 'Model0d::VariableInfo Model0d::VARIABLE_INFO')
                preClassStuff += line + '\n'
            elif in_class_init:
                if 'const size_t STATE_COUNT' in line:
                    line = line.replace('const size_t STATE_COUNT = ', 'STATE_COUNT(')
                    line = line.replace(';', '),')
                if 'const size_t VARIABLE_COUNT' in line:
                    # STATE_COUNT and VARIABLE_COUNT are declared first in the class: the other members follow them
                    line = line.replace('const size_t VARIABLE_COUNT = ', 'VARIABLE_COUNT(')
                    line = line.replace(';', '),' + memberInits + ' {')
                classInit += '    ' + line + '\n'
            else:
                # print(iL, line)
                if 'res[i] = NAN' in line:
                    line = line.replace('NAN', '0.0')

                if 'createStatesArray' in line:
                    line = line.replace('createStatesArray', 'Model0d::createStatesArray')
                if 'createVariablesArray' in line:
                    line = line.replace('createVariablesArray', 'Model0d::createVariablesArray')
                if 'deleteArray' in line:
                    line = line.replace('deleteArray', 'Model0d::deleteArray')
                
                if 'initialiseVariables' in line: 
                    line = line.replace('initialiseVariables', 'Model0d::initialiseVariables')

                    if 'initialiseVariables(' in line:
                        if 'double voi' not in line:
                            line = line.replace('(', '(double voi, ')

                    iStart = None
                    iEnd = None
                    for iL2, line2 in enumerate(lines[iL+1:], start=iL+1):
                        if line2.startswith("{"):
                            iStart = iL2
                        if line2.startswith("}"):
                            iEnd = iL2
                        if iStart is not None and iEnd is not None:
                            break
                    if iStart is None or iEnd is None:
                        print(f"generate_cpp :: ERROR: Start or end line index of initialiseVariables function not found: {iStart} {iEnd}.")
                        exit(1)

                    for iL2, line2 in enumerate(lines[iStart:iEnd], start=iStart):
                        if 'external' in line2:
                            init_lines_to_mod.append(iL2)
                
                if 'computeComputedConstants' in line:
                    line = line.replace('computeComputedConstants', 'Model0d::computeComputedConstants')
                
                # if 'computeRates' in line: 
                if 'computeRates(double ' in line:
                    line = line.replace('computeRates', 'Model0d::computeRates')
                    line = line.replace('double voi,', 'double voi, double dt,') # Bea's addition

                    iStart = None
                    iEnd = None
                    for iL2, line2 in enumerate(lines[iL+1:], start=iL+1):
                        if line2.startswith("{"):
                            iStart = iL2
                        if line2.startswith("}"):
                            iEnd = iL2
                        if iStart is not None and iEnd is not None:
                            break
                    if iStart is None or iEnd is None:
                        print(f"generate_cpp :: ERROR: Start or end line index of computeRates function not found: {iStart} {iEnd}.")
                        exit(1)

                    line = line + "\n"      # start from the original line
                    line += "{\n"           # add the opening brace
                    line += "    computeVariables(voi, dt, states, rates, variables);"  # add indented statement

                    lines_to_pass_rates.append(iStart)
                    pattern = VARIABLE_LINE_PATTERN
                    for iL2, line2 in enumerate(lines[iStart:iEnd], start=iStart):
                        # if 'variables' in line2:
                        if pattern.search(line2):
                            # print("lines to pass computeRates:", iL2, line2)
                            lines_to_pass_rates.append(iL2)

                # if 'computeVariables' in line:
                if 'computeVariables(double ' in line:
                    line = line.replace('computeVariables', 'Model0d::computeVariables')
                    line = line.replace('double voi,', 'double voi, double dt,') # Bea's addition

                    iStart = None
                    iEnd = None
                    for iL2, line2 in enumerate(lines[iL+1:], start=iL+1):
                        if line2.startswith("{"):
                            iStart = iL2
                        if line2.startswith("}"):
                            iEnd = iL2
                        if iStart is not None and iEnd is not None:
                            break
                    if iStart is None or iEnd is None:
                        print(f"generate_cpp :: ERROR: Start or end line index of computeVariables function not found: {iStart} {iEnd}.")
                        exit(1)

                    line = line + "\n"      # start from the original line
                    line += "{\n"           # add the opening brace
                    line += "    getDelayVariables(voi, dt, states, variables); // delayed values: from the past only, so first\n"
                    line += "    computeNonExternalVariables(voi, states, rates, variables);\n"
                    line += "    computeCouplingSentValues(voi, states, rates, variables);\n"
                    line += "    getExternalVariables(voi, dt, states, variables);\n"
                    line += "    computeCouplingDependentVariables(voi, states, rates, variables);\n"
                    # line += "    }\n"  
                    line += "}\n"  

                    lines_to_pass_variables.append(iStart)
                    pattern = VARIABLE_LINE_PATTERN
                    for iL2, line2 in enumerate(lines[iStart:iEnd], start=iStart):
                        # if 'variables' in line2:
                        if pattern.search(line2):
                            # print("lines to pass computeVariables:", iL2, line2)
                            lines_to_pass_variables.append(iL2)
                    lines_to_pass_variables.append(iEnd)
                    
                    line += "\n"
                    line += "void Model0d::computeNonExternalVariables(double voi, double *states, double *rates, double *variables)\n"
                    line += "{\n"
                    # lines_before, lines_after and coupling_send_code: see the analysis after gen.setModel()
                    for line2 in lines_before:
                        line += line2+"\n"
                    line += "}\n"
                    line += "\n"
                    line += "void Model0d::computeCouplingDependentVariables(double voi, double *states, double *rates, double *variables)\n"
                    line += "{\n"
                    for line2 in lines_after:
                        line += line2+"\n"
                    line += "}\n"
                    line += "\n"
                    # values sent to the 1d solver by each 1d-0d connection, in couplingSend (see analyse_coupling_connections())
                    line += "void Model0d::computeCouplingSentValues(double voi, double *states, double *rates, double *variables)\n"
                    line += "{\n"
                    line += coupling_send_code
                    line += "}" 

                if iL in init_lines_to_mod:
                    idx0 = line.find("external")
                    idx1 = line.find(";")
                    line = line.replace(line[idx0:idx1+1], '0.0; // external coupling variable computed at initial time step iteration')

                if 'ExternalVariable externalVariable' in line:
                    line = line.replace(', ExternalVariable externalVariable', '')

                if 'voi' in line:
                    line = line.replace('void', 'XXX')
                    # line = line.replace('voi', 'voi_') # I dont like this format:))
                    line = line.replace('voi', 'voiLoc')
                    line = line.replace('XXX', 'void')
                if 'dt' in line:
                    line = line.replace('dt', 'dtLoc')
                if 'states' in line:
                    # line = line.replace('states', 'states_') # I dont like this format:))
                    line = line.replace('states', 'statesLoc')
                if 'rates' in line:
                    # line = line.replace('rates', 'rates_') # I dont like this format:))
                    line = line.replace('rates', 'ratesLoc')
                if 'variables' in line:
                    # line = line.replace('variables', 'variables_') # I dont like this format:))
                    line = line.replace('variables', 'varLoc')

                # postClassInit += line + '\n'
                if (iL not in lines_to_pass and iL not in lines_to_pass_rates and iL not in lines_to_pass_variables):
                    postClassInit += line + '\n'
                else:
                    pass

        classInit += f"""
            states = createStatesArray();
            rates = createStatesArray(); // same size as states, should really be different function
            variables = createVariablesArray();
        """

        classInit += """
        }

        Model0d::~Model0d() {
            // Clean up after ourselves.
            deleteArray(states);
            deleteArray(rates);
            deleteArray(variables);
        """

        if self.solver == 'CVODE':
            classInit += """
            delete userData;
            SUNLinSolFree(linearSolver);
            SUNMatDestroy(matrix);
            N_VDestroy_Serial(y);
            CVodeFree(&solver);
            SUNContext_Free(&context);   
        """
        elif self.solver == 'PETSC':
            classInit += """
            delete userData;
            if (y) { VecDestroy(&y); y = nullptr; }
            if (rates_vec) { VecDestroy(&rates_vec); rates_vec = nullptr; }
            if (J) { MatDestroy(&J); J = nullptr; }
            if (ts) { TSDestroy(&ts); ts = nullptr; }
            // if (snes) SNESDestroy(&snes); // this is destroyed by TSDestroy
            // if (ksp) KSPDestroy(&ksp); // this is destroyed by TSDestroy
            // if (pc) PCDestroy(&pc); // this is destroyed by TSDestroy   
        """
        else:
            classInit += """
            deleteArray(states0);
            deleteArray(k1);
            deleteArray(k2);
            deleteArray(k3);
            deleteArray(k4);
        """

        #XXX the buffers for the MPI coupling with the 1d solver are std::vector type, so their memory is freed automatically

        classInit += """
        }
        """

        # print("#################################################")
        # print("preClassStuff")
        # print(preClassStuff)
        # print("#################################################")
        # print("classInit")
        # print(classInit)
        # print("#################################################")
        # print("postClassInit")
        # print(postClassInit) 


        solverInitFunction = """ 
void Model0d::set_ode_solver(std::string ODEsolver)
{
    solver_0d = ODEsolver;
    """

        if self.solver == 'CVODE':
            solverInitFunction += f"""
    // Create our SUNDIALS context object.
    SUNContext_Create(SUN_COMM_NULL, &context);
    // Create our CVODE solver.
    solver = CVodeCreate(CV_BDF, context);
    // Initialise our CVODE solver.
    y = N_VMake_Serial(STATE_COUNT, states, context);
    CVodeInit(solver, func, voi, y);
    // Set our user data.
    userData = new UserOdeData(variables, std::bind(&Model0d::computeRates, this, std::placeholders::_1, 
                                                    std::placeholders::_2, std::placeholders::_3, 
                                                    std::placeholders::_4, std::placeholders::_5), solver);
    CVodeSetUserData(solver, userData);
    // Set our maximum number of internal steps (default 500).
    long int mxsteps = {self.nMaxSteps};
    CVodeSetMaxNumSteps(solver, mxsteps);
    // Set our maximum absolute step size.
    sunrealtype hmax = {self.dtSolver};
    CVodeSetMaxStep(solver, hmax);
    // Set our linear solver.
    // Create matrix object.
    matrix = SUNDenseMatrix(STATE_COUNT, STATE_COUNT, context);
    // Create linear solver object.
    linearSolver = SUNLinSol_Dense(y, matrix, context);
    // Attach linear solver module
    CVodeSetLinearSolver(solver, linearSolver, matrix);
    // Set our scalar relative and absolute tolerances.
    sunrealtype reltol = {REL_TOL}; // TODO get this from user_inputs.yaml too
    sunrealtype abstol = {ABS_TOL}; // TODO get this from user_inputs.yaml too
    CVodeSStolerances(solver, reltol, abstol); 
"""
        elif self.solver == 'PETSC':
            solverInitFunction += f"""
    // Set up PETSc TS ODE solver (fixed time steps, only shortened to retry a failed step). Every PETSc call is
    // checked: PETSC_CHECK stops the program if it fails.

    // Create solution vector
    PETSC_CHECK(VecCreate(PETSC_COMM_WORLD, &y));
    PETSC_CHECK(VecSetSizes(y, PETSC_DECIDE, STATE_COUNT));
    PETSC_CHECK(VecSetFromOptions(y));

    // Create TS
    PETSC_CHECK(TSCreate(PETSC_COMM_WORLD, &ts)); // create the timestepper (TS) object
    PETSC_CHECK(TSSetProblemType(ts, TS_NONLINEAR)); // TSProblemType is one of TS_LINEAR or TS_NONLINEAR
    PETSC_CHECK(TSSetRHSFunction(ts, NULL, TSRHSfunc, this)); // set the RHS function
    // 'this' --> pointer to your Model0d object; PETSc stores it, so that every time it calls TSRHSfunc, it passes that same pointer as ctx.
    PETSC_CHECK(TSSetSolution(ts, y));


#ifdef SOLVER_CN
    PETSC_CHECK(TSSetType(ts, TSCN)); // set the solution method: Crank-Nicolson
    PETSC_CHECK(TSSetTimeStep(ts, {self.dtSolver})); // set the initial time step
    PETSC_CHECK(TSSetMaxSteps(ts, std::numeric_limits<PetscInt>::max())); // maximum number of time steps: no limit
    // PETSC_CHECK(TSSetMaxTime(ts, 10.));
    // No other options needed
#endif

#ifdef SOLVER_BDF1
    PETSC_CHECK(TSSetType(ts, TSBDF)); // set the solution method: backward differentiation formula (BDF)
    PETSC_CHECK(TSBDFSetOrder(ts, 1));
    PETSC_CHECK(TSSetTimeStep(ts, {self.dtSolver})); // set the initial time step
    PETSC_CHECK(TSSetMaxSteps(ts, std::numeric_limits<PetscInt>::max())); // maximum number of time steps: no limit
    // PETSC_CHECK(TSSetMaxTime(ts, 10.));

    // Fixed time steps (the default TSBDF adaptor would change them)
    PETSC_CHECK(PetscOptionsSetValue(NULL, "-ts_adapt_type", "none"));
    // REMOVED: Options for matrix-free BDF1
    // PetscOptionsSetValue(NULL, "-ts_bdf_single_jacobian", NULL);
    // PetscOptionsSetValue(NULL, "-snes_mf_operator", NULL);
#endif

#ifdef SOLVER_BDF2
    PETSC_CHECK(TSSetType(ts, TSBDF)); // set the solution method: backward differentiation formula (BDF)
    PETSC_CHECK(TSBDFSetOrder(ts, 2));
    PETSC_CHECK(TSSetTimeStep(ts, {self.dtSolver})); // set the initial time step
    PETSC_CHECK(TSSetMaxSteps(ts, std::numeric_limits<PetscInt>::max())); // maximum number of time steps: no limit
    // PETSC_CHECK(TSSetMaxTime(ts, 10.));

    // Fixed time steps (the default TSBDF adaptor would change them)
    PETSC_CHECK(PetscOptionsSetValue(NULL, "-ts_adapt_type", "none"));
    // REMOVED: Options for matrix-free BDF2
    // PetscOptionsSetValue(NULL, "-ts_bdf_single_jacobian", NULL);
    // PetscOptionsSetValue(NULL, "-snes_mf_operator", NULL);
#endif

    // Failed time steps: retried with shorter steps, as CVODE does. The nonlinear system of an implicit step can have
    // no solution when the rates jump inside the step, e.g. when a phase variable of the heart module crosses an
    // integer (start of a heart cycle): the Newton iterations then fail whatever the Jacobian or the tolerances.
    // After a failed Newton solve, PETSc retries the step with a time step 4 times shorter (-ts_adapt_scale_solve_failed).
    // By default, PETSc stops at the first failure. The number of failures is not limited, because PETSc counts them
    // over the whole run (the counter is only reset at step 0): the limit is set per step, by TSSetMaxStepRejections.
    PETSC_CHECK(TSSetMaxSNESFailures(ts, PETSC_UNLIMITED)); // maximum number of failed Newton solves in the whole run: no limit
    // The program stops (DIVERGED_STEP_REJECTED) after 10 retries of the same step (smallest step ~ dt*4^-10).
    PETSC_CHECK(TSSetMaxStepRejections(ts, 10)); // maximum number of retries of each step (PETSc default, set explicitly)
    // The last of the shortened steps ends exactly at the end of the solver step (TSSetMaxTime in solveOneStep), so
    // the output times and the coupling times stay exact. The default would step over it.
    PETSC_CHECK(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP)); // shorten the last step to end exactly at the final time

    // Tolerances: with fixed time steps, they are used by the convergence test of the Newton iterations (SNESConvergedWRMS)
    // in order: scalar absolute tolerances, vector of absolute tolerances, scalar relative tolerances, vector of relative tolerances
    double reltol = {REL_TOL}; // TODO get this from user_inputs.yaml too
    double abstol = {ABS_TOL}; // TODO get this from user_inputs.yaml too
    PETSC_CHECK(TSSetTolerances(ts, abstol, NULL, reltol, NULL));

    // Nonlinear solver
    PETSC_CHECK(TSGetSNES(ts, &snes)); // return the SNES (nonlinear solver) associated with the TS (timestepper) context (valid only for nonlinear problems)
    PETSC_CHECK(SNESSetType(snes, SNESNEWTONLS)); // set the algorithm/method to be used to solve the nonlinear system with the given SNES
    // No line search: full Newton steps, as in CVODE. SNESNEWTONLS is Newton's method with a line search object, and
    // its type SNESLINESEARCHBASIC (also called SNESLINESEARCHNONE) is "not a line search at all" (PETSc docs): it takes
    // the full step x_new = x - lambda*dx with lambda = 1 fixed, and only evaluates the residual at x_new for the
    // convergence test. The default type (bt, backtracking) would instead shrink lambda until the residual decreases:
    // near the solution, the residual of stiff models is dominated by round-off and cannot decrease, so the steps
    // were shrunk to ~1e-12 and the Newton iterations failed.
    SNESLineSearch linesearch;
    PETSC_CHECK(SNESGetLineSearch(snes, &linesearch));
    PETSC_CHECK(SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC));
    // Set the function (SNESConvergedWRMS) that is to be used to test for convergence of the nonlinear iterative solution.
    PETSC_CHECK(SNESSetConvergenceTest(snes, SNESConvergedWRMS, ts, NULL)); // see SNESConvergedWRMS

    // Linear solver (dense)
    PETSC_CHECK(SNESGetKSP(snes, &ksp)); // return the KSP (linear solver) context for the SNES solver
    // The Jacobian is built by PETSc with finite differences, one column at a time (no Jacobian function is given),
    // and the linear systems are solved exactly with a dense LU factorisation, as in CVODE. 
    // These systems are small but badly scaled (stiff modes): iterative solvers (GMRES with ILU or Jacobi) would give inexact Newton steps that fail.
    // PETSC_CHECK(KSPSetType(ksp, KSPGMRES)); // GMRES
    // PETSC_CHECK(KSPSetType(ksp, KSPFGMRES)); // flexible GMRES
    PETSC_CHECK(KSPSetType(ksp, KSPPREONLY)); // KSPPREONLY: method that applies ONLY the preconditioner exactly once.

    // Preconditioner
    PETSC_CHECK(KSPGetPC(ksp, &pc)); // return a pointer to the preconditioner (PC) context with the KSP

    // Preconditioner: exact LU factorisation, for every method (KSPPREONLY applies it once, i.e. a direct solve)
    // PETSC_CHECK(PCSetType(pc, PCNONE));
    // PETSC_CHECK(PCSetType(pc, PCJACOBI)); // PCJACOBI: Jacobi, i.e. diagonal scaling preconditioning.
    // PETSC_CHECK(PCSetType(pc, PCILU));
    PETSC_CHECK(PCSetType(pc, PCLU)); // PCLU: this uses a direct solver, based on LU factorization, as a preconditioner.

    PETSC_CHECK(TSSetFromOptions(ts)); // set various TS parameters from the options database
    PETSC_CHECK(TSSetUp(ts)); // set up the internal data structures for the later use of TS

    // User data
    userData = new UserOdeData(variables, std::bind(&Model0d::computeRates, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5));
    userData->tOld = voi;

"""
        else:
            solverInitFunction += """
    states0 = createStatesArray(); // store initial state, before evolving to the next time level
    k1 = createStatesArray(); // same size as states, should really be different function
    k2 = createStatesArray(); // same size as states, should really be different function
    k3 = createStatesArray(); // same size as states, should really be different function
    k4 = createStatesArray(); // same size as states, should really be different function
    // Runge-Kutta 4 weights
    wRK4[0] = 1./6.;
    wRK4[1] = 1./3.;
    wRK4[2] = 1./3.;
    wRK4[3] = 1./6.;
"""
        solverInitFunction += """
}
"""
        
        # print("#################################################")
        # print("solverInitFunction")
        # print(solverInitFunction) 

        # functions handling the MPI coupling with the 1d solver (through the coupler)
        couplingFunctions = ''
        if self.couple_to_1d and N1d0dTot>0:

            iStart = None
            iEnd = None
            for iL, line in enumerate(lines_tmp):
                if "Functions handling the MPI coupling" in line:
                    iStart = iL
                    break
            for iL, line in enumerate(lines_tmp[iStart+1:], start=iStart+1):
                if "End functions" in line:
                    iEnd = iL
                    break
            for iL, line in enumerate(lines_tmp[iStart:iEnd+1], start=iStart):
                couplingFunctions += line #+ '\n'
            couplingFunctions += '\n'

            # 1d-0d connections compiled into this model, as written in its connections file (<model>_coupler1d0d.json):
            # initCoupling() checks them against the file read by the coupler (see COUPLING_CONN_FIELDS)
            couplingFunctions += "std::vector<std::map<std::string, int>> Model0d::compiledConnections() const\n{\n"
            couplingFunctions += "    // 1d-0d connections of this model at generation, as in its connections file (<model>_coupler1d0d.json)\n"
            couplingFunctions += "    return {\n"
            for iC in range(N1d0dTot):
                fields = ", ".join(f'{{"{key}", {value}}}' for key, value in self.conn_1d_0d_info[str(iC+1)].items()
                                   if isinstance(value, int) and not isinstance(value, bool))
                couplingFunctions += f"        {{{fields}}}, // connection {iC+1}\n"
            couplingFunctions += "    };\n}\n\n"


        # print("#################################################")
        # print("couplingFunctions")
        # print(couplingFunctions)

        outputFunctions = ''
        iStart = None
        iEnd = None
        for iL, line in enumerate(lines_tmp):
            if "Functions for import/export of model outputs" in line:
                iStart = iL
                break
        for iL, line in enumerate(lines_tmp[iStart+1:], start=iStart+1):
            if "End functions" in line:
                iEnd = iL
                break
        for iL, line in enumerate(lines_tmp[iStart:iEnd+1], start=iStart):
            if "VARIABLE_INFO[i].type == EXTERNAL ||" in line:
                # external variables are written to the output in coupled models (values received from the 1d solver)
                # and with delay variables (delayed values)
                if self.couple_to_1d or delay_variables:
                    outputFunctions += line + '\n'
                else:
                    line = line.replace("VARIABLE_INFO[i].type == EXTERNAL || ", "")
                    outputFunctions += line #+ '\n'
            else:
                outputFunctions += line #+ '\n'
        outputFunctions += '\n'


        # print("#################################################")
        # print("outputFunctions")
        # print(outputFunctions)


        # # find the function for computeVariables to create a 
        # # compute non_external variables function
        # non_external_variables_function = \
        #         'void Model0d::computeNonExternalVariables(double voi_, double *states_, double *rates_, double *variables_) \n'
        # startFunction = False
        # for line in gen.implementationCode().split('\n'):
        #     if line.startswith('void computeVariables'):
        #         startFunction = True
        #     if startFunction:
        #         if line.startswith('}'):
        #             non_external_variables_function += line + '\n'
        #             break
        #         if 'externalVariable' in line:
        #             continue
        #         if 'voi' in line:
        #             line = line.replace('void', 'XXX')
        #             line = line.replace('voi', 'voi_')
        #             line = line.replace('XXX', 'void')
        #         if 'states' in line:
        #             line = line.replace('states', 'states_')
        #         if 'rates' in line:
        #             line = line.replace('rates', 'rates_')
        #         if 'variables' in line:
        #             line = line.replace('variables', 'variables_')
        #         non_external_variables_function += line + '\n'


#         # and generate a function to compute external variables #XXX Finbar's old code
#         computeEV = f""" 
# double Model0d::externalVariable(double voi_, double *states_, double *rates_, double *variables_, size_t index)
# {{
#     // if voi is zero we may need to calculate the non_external_variables
#     // because they can be needed for the external variables
#     if (voi_ == 0.0) {{
#         computeNonExternalVariables(voi_, states_, rates_, variables_);
#     }}
#         """
#         for ext_variable in external_variable_info:
#             print(ext_variable.keys())

#             # variable or state index
#             if ext_variable["state_or_variable"] == "state":
#                 state_index = ext_variable['state_index']
#             else:
#                 variable_index = ext_variable['variable_index']

#             # TODO Finbar: for each external variable, include a label for its corresponding port
#             # variable, ie when we need qBC, we give P_C and possibly R_T1.
#             # TODO Finbar: create a config file which details the types of external variables we 
#             # can have and their corresponding port variables. 
#             # a new one will be created for each type of coupling.

#             # TODO if time_dof_0d = 0 then we don't store the variables in the buffer, only store when it equals -1
#             # set up so that when time_dof_0d is not -1 we don't step forward in the buffer, we only interpolate 
#             # from it, we only step forward in the buffer when time_dof_0d is -1
#             # use a time_0d variable to check where voi is in terms of time_0d
#             # TODO we could run a rough CVODE where we only save at time_output points, then linearly interpolate...
#             #  this would be quite easy to implement.. unsure about stability.
#             if ext_variable['variable_type'] == 'delay':
                    
#                 delay_variable_index = ext_variable['delay_variable_index']
#                 delay_amount_index = ext_variable['delay_amount_index']
#                 computeEV += f'  double dt_fraction = (voi_ - voi) / dt;\n'
#                 computeEV += f'  double value;\n'
#                 computeEV += f'  if (index == {delay_variable_index}) {{\n'
#                 computeEV += f'    if (voi_ < variables_[{delay_amount_index}]) {{\n'
#                 computeEV += f'      if (buffersInitialised != true) {{return 0.0;}};\n'
#                 computeEV += f'      if (time_dof_0d == -1) {{;\n'
#                 if ext_variable["state_or_variable"] == "state":
#                     computeEV += f'        storeBufferVariable({delay_variable_index}, states_[{state_index}]);\n'
#                 else:
#                     computeEV += f'        storeBufferVariable({delay_variable_index}, variables_[{variable_index}]);\n'
#                 computeEV += f'      }};\n'
#                 computeEV += f'      return 0.0;\n'
#                 computeEV += f'    }} else {{;\n'
#                 computeEV += f'      if (time_dof_0d == -1) {{;\n'
#                 computeEV += f'        value = getBufferVariable({delay_variable_index}, 1.0, 1);\n'
#                 computeEV += f'      }} else {{;\n'
#                 computeEV += f'        value = getBufferVariable({delay_variable_index}, dt_fraction, 0);\n'
#                 computeEV += f'      }}\n'
#                 computeEV += f'      // save the current value of the variable to the circle buffer\n'
#                 computeEV += f'      if (time_dof_0d == -1) {{;\n'
#                 if ext_variable["state_or_variable"] == "state":
#                     computeEV += f'        storeBufferVariable({delay_variable_index}, states_[{state_index}]);\n'
#                 else:
#                     computeEV += f'        storeBufferVariable({delay_variable_index}, variables_[{variable_index}]);\n'
#                 computeEV += f'      }};\n'
#                 computeEV += f'      return value;\n'
#                 computeEV += f'    }};\n'
#                 computeEV += f'  }}\n'

#             elif ext_variable['variable_type'] == 'input':
#                 if ext_variable['state_or_variable'] == 'state':
#                     print('Input BC variable can not be a state variable, exiting')
#                     exit()
                    
#                 elif ext_variable['state_or_variable'] == 'variable':
#                     if ext_variable['coupled_to_type'] == 'FV_1d':
#                         computeEV += f'  if (index == {variable_index}) {{\n'
#                         if ext_variable['flow_or_pressure_bc'] == 'flow':
#                             computeEV += f'    int vessel1d_idx = cellml_index_to_vessel1d_info["{variable_index}"]["vessel1d_idx"];\n'
#                             computeEV += f'    int inlet0_or_outlet1_bc = cellml_index_to_vessel1d_info["{variable_index}"]["bc_inlet0_or_outlet1"];\n'
#                             if 'port_variable_index' in ext_variable.keys():
#                                 port_variable_index = ext_variable['port_variable_index']
#                                 if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
#                                     control_variable_index = ext_variable['control_variable_index']
#                                     computeEV += f'    double value = get_model1d_flow(vessel1d_idx, variables_[{port_variable_index}], inlet0_or_outlet1_bc, variables_[{control_variable_index}]);\n'
#                                 else:
#                                     computeEV += f'    double value = get_model1d_flow(vessel1d_idx, variables_[{port_variable_index}], inlet0_or_outlet1_bc);\n' 
#                             elif 'port_state_index' in ext_variable.keys():
#                                 port_state_index = ext_variable['port_state_index']
#                                 if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
#                                     control_variable_index = ext_variable['control_variable_index']
#                                     computeEV += f'    double value = get_model1d_flow(vessel1d_idx, states_[{port_state_index}], inlet0_or_outlet1_bc, variables_[{control_variable_index}]);\n'
#                                 else:
#                                     computeEV += f'    double value = get_model1d_flow(vessel1d_idx, states_[{port_state_index}], inlet0_or_outlet1_bc);\n'
#                         elif ext_variable['flow_or_pressure_bc'] == 'pressure':
#                             computeEV += f'    int vessel1d_idx = cellml_index_to_vessel1d_info["{variable_index}"]["vessel1d_idx"];\n'
#                             computeEV += f'    int inlet0_or_outlet1_bc = cellml_index_to_vessel1d_info["{variable_index}"]["bc_inlet0_or_outlet1"];\n'
#                             if 'port_variable_index' in ext_variable.keys():
#                                 port_variable_index = ext_variable['port_variable_index']
#                                 computeEV += f'    double value = get_model1d_pressure(vessel1d_idx, variables_[{port_variable_index}], inlet0_or_outlet1_bc);\n' 
#                             elif 'port_state_index' in ext_variable.keys():
#                                 port_state_index = ext_variable['port_state_index']
#                                 computeEV += f'    double value = get_model1d_pressure(vessel1d_idx, states_[{port_state_index}], inlet0_or_outlet1_bc);\n'
#                             print("Currently we don't implement the pressure coupling. ",
#                                   "As it would require solving a DAE in CellML")
#                             exit()
#                         computeEV += f'    return value;\n'
#                         computeEV += f'  }}\n'

#         computeEV += f"""
# return 0.0;
# }}
#         """


        # generate a function to compute external variables #XXX Bea's new code

        # TODO Finbar: for each external variable, include a label for its corresponding port
        # variable, ie when we need qBC, we give P_C and possibly R_T1.
        # TODO Finbar: create a config file which details the types of external variables we 
        # can have and their corresponding port variables. 
        # a new one will be created for each type of coupling.
        # TODO Beatrice: check compatibility with Finbar's code for delay variables

        # Delay variables (see delay_variables above): each delayed value is interpolated in the history of the variable
        # to delay (DelayHistory), which holds its values at the end of the time steps, whatever their length (fully 0d
        # and coupled runs). It depends only on past values, so it is set before the other variables are computed.
        computeEV = """
void Model0d::getDelayVariables(double voiLoc, double dtLoc, double *statesLoc, double *varLoc)
{
    // delayed value = value of the variable to delay at voiLoc - delay, interpolated in its history
"""
        for k, d in enumerate(delay_variables):
            computeEV += (f"    varLoc[{d['delayed_index']}] = delayHistories[{k}].valueAt(voiLoc - varLoc[{d['amount_index']}]);"
                          f" // {d['name']}\n")
        computeEV += """}

void Model0d::initDelayHistories()
{
    // history of each variable to delay, started with its initial value: the delayed value is the initial value of the
    // variable to delay for t < delay
"""
        if delay_variables:
            computeEV += f"    delayHistories.assign({len(delay_variables)}, DelayHistory());\n"
        for k, d in enumerate(delay_variables):
            source = d['source']
            computeEV += f"""    // {d['name']}
    if (!(variables[{d['amount_index']}] > 0.0)) {{
        std::cerr << "0d solver :: ERROR: delay variable {d['name']}: the delay must be > 0 ("
                  << variables[{d['amount_index']}] << " given)" << std::endl;
        exit(1);
    }}
    if (variables[{d['amount_index']}] < dt) {{
        std::cout << "0d solver :: WARNING: delay variable {d['name']}: the delay " << variables[{d['amount_index']}]
                  << " is shorter than the time step " << dt << ": the delayed value lags by up to one time step" << std::endl;
    }}
    delayHistories[{k}].reset(voi, {source});
    variables[{d['delayed_index']}] = {source};
"""
        computeEV += """}

void Model0d::storeDelayHistories()
{
    // at the end of each time step: value of each variable to delay, and drop the values no longer needed
"""
        for k, d in enumerate(delay_variables):
            computeEV += f"    delayHistories[{k}].add(voi, {d['source']});\n"
            computeEV += f"    delayHistories[{k}].dropBefore(voi - variables[{d['amount_index']}]);\n"
        computeEV += """}
"""

        computeEV += """
void Model0d::getExternalVariables(double voiLoc, double dtLoc, double *statesLoc, double *varLoc)
{
    // Next, deal with 1d-0d coupling
"""

        if self.couple_to_1d:
            computeEV += """
    if (N1d0d>0) {
        // Step 1: send data
        // S1: compute and send time & internal time step data to coupler
        zero_data_dt[0] = voiLoc;
        zero_data_dt[1] = dtLoc;
        MPI_Send(zero_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DT, MPI_COMM_WORLD);

        // then the values of all the 1d-0d connections, computed by computeCouplingSentValues():
        // {pressure P0D (or flow), resistance R} for each connection (one message, one row per connection)
        for (int i = 0; i < N1d0d; ++i) {
            zero_data[i*DATA_LENGTH+0] = couplingSend[2*i];
            zero_data[i*DATA_LENGTH+1] = couplingSend[2*i+1];
        }
"""
            for i, (conn, plan) in enumerate(zip(coupling_connections, coupling_plans)):
                # values sent to the 1d solver (computeCouplingSentValues()), checked for every connection: a pressure
                # P0D and a resistance R (flow bc, or pressure bc exchanged as a flow bc), or a flow (pressure bc, R = 0)
                if conn['flow_bc'] or plan['converted']:
                    computeEV += f"""        if (!std::isfinite(couplingSend[{2*i}]) || !std::isfinite(couplingSend[{2*i+1}]) || couplingSend[{2*i+1}] < 0.0) {{
            std::cerr << "0d solver :: ERROR: {conn['where']}: the values sent to the 1d solver, pressure "
                      << couplingSend[{2*i}] << " and resistance " << couplingSend[{2*i+1}] << ", at time " << voiLoc
                      << ", must be finite, with a non-negative resistance. Exiting." << std::endl;
            exit(1);
        }}
"""
                else:
                    computeEV += f"""        if (!std::isfinite(couplingSend[{2*i}])) {{
            std::cerr << "0d solver :: ERROR: {conn['where']}: the flow sent to the 1d solver, " << couplingSend[{2*i}]
                      << ", at time " << voiLoc << ", is not finite. Exiting." << std::endl;
            exit(1);
        }}
"""
            computeEV += """        MPI_Send(zero_data.data(), N1d0d*DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DATA, MPI_COMM_WORLD);
        // ...wait to receive back...
        // Step 2: receive the values computed by the 1d solver for all the 1d-0d connections
        MPI_Recv(parent_data.data(), N1d0d*DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // stop if a value received from the 1d solver is not finite (inf or NaN), for any 0d solver, instead of
        // continuing with it
"""
            for i, conn in enumerate(coupling_connections):
                computeEV += f"""        if (!std::isfinite(parent_data[{i}*DATA_LENGTH+0])) {{
            std::cerr << "0d solver :: ERROR: {conn['where']}: the value received from the 1d solver is not finite ("
                      << parent_data[{i}*DATA_LENGTH+0] << ") at time " << voiLoc << " (time step " << dtLoc << "). Exiting."
                      << std::endl;
            exit(1);
        }}
"""
            for i, (conn, plan) in enumerate(zip(coupling_connections, coupling_plans)):
                if plan['converted']:
                    computeEV += (f"        varLoc[{conn['received']}] = couplingSend[{2*i}]{'+' if conn['sign'] > 0 else '-'}"
                                  f"couplingSend[{2*i+1}]*parent_data[{i}*DATA_LENGTH+0]; // connection {conn['key']}: "
                                  "pressure at the boundary of the 1d vessel, from the flow received (pressure bc "
                                  "exchanged as a flow bc)\n")
                else:
                    computeEV += (f"        varLoc[{conn['received']}] = parent_data[{i}*DATA_LENGTH+0]; // connection "
                                  f"{conn['key']}\n")
            computeEV += """        // Step 3: now exit this function and compute rates
    }
"""
        computeEV += """
}
"""
        
        computeEV += """
void Model0d::getExternalVariablesVolume(double *varLoc)
{
    // Values of the global connections (e.g. the 1d volume, added to the 0d volume sum to track the total volume in the
    // system), received from the 1d solver at start-up (initialiseVariablesAndComputeConstants()) and at the end of each
    // time step, not for the intermediate/internal time steps taken by the solver: one message, one row per global
    // connection (see "Global connections" in coupler_mpi_common.h)
"""
        global_conns = global_coupling_connections(self.conn_1d_0d_info) if self.couple_to_1d else []
        if global_conns:
            computeEV += """
    MPI_Recv(parent_data_vol.data(), N1d0dGlobal*DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_VOL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
"""
            for row, (key, conn) in enumerate(global_conns):
                # the only type of global connection so far: the volume sum, {total blood volume of the 1d model [m3], 0}
                if conn["cellml_idx"] < 0:
                    print(f"generate_cpp :: ERROR: 1d-0d connection {key} (volume sum): the 0d variable that receives the 1d "
                          "volume was not found in the analysed model")
                    exit(1)
                computeEV += f"""    if (!std::isfinite(parent_data_vol[{row}*DATA_LENGTH+0])) {{
        std::cerr << "0d solver :: ERROR: 1d-0d connection {key} (volume sum): the 1d blood volume received from the 1d "
                  << "solver is not finite (" << parent_data_vol[{row}*DATA_LENGTH+0] << ") at time " << voi << ". Exiting."
                  << std::endl;
        exit(1);
    }}
    varLoc[{conn["cellml_idx"]}] = parent_data_vol[{row}*DATA_LENGTH+0]; // connection {key} (volume sum): 1d blood volume
"""

        computeEV += """
}
"""

        # print("#################################################")
        # print("computeEV")
        # print(computeEV)


#         if self.couple_to_1d: #XXX Finbar's old code
#             # external interaction functions
#             externalInteractionFunctions = """
# void Model0d::connect_to_model1d(Model1d* model1d){
#     model1d_ptr = model1d; 
# }
#             """
#             if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
#                 externalInteractionFunctions += """
# double Model0d::get_model1d_flow(int vessel_idx, double p_C, int input0_output1_bc, double R_T1_wCont){
#                 """
#             else:
#                 externalInteractionFunctions += """
# double Model0d::get_model1d_flow(int vessel_idx, double p_C, int input0_output1_bc){
#                 """
#             externalInteractionFunctions += """
#     // get the flow from the 1d model

#     // get qBC and step forward the fluctuation
#             """
            
#             if self.solver == 'RK4':
#                 externalInteractionFunctions += """
#     int add_to_fluct = 1;
#     double time_int_weight = 0.0;
#     double time_0d = 0.0;
#     if (time_dof_0d == 0) {
#         time_int_weight = 1.0/6.0;
#         time_0d = voi;
#     }
#     else if (time_dof_0d == 1 || time_dof_0d == 2) {
#         time_int_weight = 1.0/3.0;
#         time_0d = voi + dt/2.0;
#     }
#     else if (time_dof_0d == 3) {
#         time_int_weight = 1.0/6.0;
#         time_0d = voi + dt;
#     }
#     else if (time_dof_0d == -1) {
#         time_int_weight = 0.0;
#         add_to_fluct = 0;
#         time_0d = voi + dt;
#     }
#     else {
#         std::cout << "time_dof_0d is not set correctly" << std::endl;
#         exit(1);
#     }
#             """

#             if 'control_variable_index' in ext_variable.keys() and ext_variable['control_variable_index'] != None:
#                 externalInteractionFunctions += """
#     double flow = model1d_ptr->getqBCAndAddToFluctWrap(vessel_idx, add_to_fluct, 
#                                                 time_0d, time_int_weight, 
#                                                 p_C, input0_output1_bc, R_T1_wCont)/pow(10,6); 
#     // divide by 10^6 to get m3/s from ml/s
#     return flow;
# }
#                 """
#             else:
#                 externalInteractionFunctions += """
#     double flow = model1d_ptr->getqBCAndAddToFluctWrap(vessel_idx, add_to_fluct, 
#                                                 time_0d, time_int_weight, 
#                                                 p_C, input0_output1_bc)/pow(10,6); 
#     // divide by 10^6 to get m3/s from ml/s
#     return flow;
# }
#                 """
                
#             # externalInteractionFunctions += """
                
#             #     // below is what I did previously
#             #     // double * all_vals = (double *) malloc(10*sizeof(double));
#             #     // if (input0_output1_bc == 0){
#             #     //     // TODO Finbar, should be evalSolSpaceTime, where the time is an input and 
#             #     //     // it interpolates the solution
#             #     //     model1d_ptr->evalSol(vessel_idx, 0.0, 0, all_vals); 
#             #     // } else {
#             #     //     double xSample = model1d_ptr->vess[vessel_idx].L;
#             #     //     int iSample = model1d_ptr->vess[vessel_idx].NCELLS - 1;
#             #     //     model1d_ptr->evalSol(vessel_idx, xSample, iSample, all_vals); 
#             #     // }
#             #     // double flow = all_vals[1]/pow(10,6); // index 1 is the flow, divide by 10^6 to get m3/s from ml/s
#             #     // // TODO make the conversion of units more generic
#             #     // // std::cout << "flow: " << std::scientific << flow << std::endl;
#             #     return flow; // 
#             # } 
#             # """

#             # Currently we don't implement the pressure coupling
#             # double Model0d::get_model1d_pressure(int vessel_idx, double qBC, int input0_output1_bc){
#             #     // get the pressure from the 1d model
#             # """ 
            
#             # if self.solver == 'RK4':
#             #     externalInteractionFunctions += """
                
#             #     int add_to_fluct = 1;
#             #     double time_int_weight = 0.0;
#             #     double time_0d = 0.0;
#             #     if (time_dof_0d == 0) {
#             #         time_int_weight = 1.0/6.0;
#             #         time_0d = voi;
#             #     }
#             #     else if (time_dof_0d == 1 || time_dof_0d == 2) {
#             #         time_int_weight = 1.0/3.0;
#             #         time_0d = voi + dt/2.0;
#             #     }
#             #     else if (time_dof_0d == 3) {
#             #         time_int_weight = 1.0/6.0;
#             #         time_0d = voi + dt;
#             #     }
#             #     else if (time_dof_0d == -1) {
#             #         time_int_weight = 0.0;
#             #         add_to_fluct = 0;
#             #         time_0d = voi + dt;
#             #     }
#             #     else {
#             #         std::cout << "time_dof_0d is not set correctly" << std::endl;
#             #         exit(1);
#             #     }
#             # """

#             # externalInteractionFunctions += """
#             #     double pressure = model1d_ptr->getPressureAndAddToFluct(vessel_idx, add_to_fluct, 
#             #                                                 time_0d, time_int_weight, 
#             #                                                 qBC, input0_output1_bc)/10; 
#             #     # divide by 10 to go from dynes/cm^2 to Pa
#             #     return pressure;

#             #     // below is what I did previously
#             #     // double * all_vals = (double *) malloc(8*sizeof(double));
#             #     // if (input0_output1_bc == 0){
#             #     //     model1d_ptr->sampleMid(vessel_idx, all_vals, 0.0); 
#             #     // } else {
#             #     //     double xSample = model1d_ptr->vess[vessel_idx].L;
#             #     //     model1d_ptr->sampleMid(vessel_idx, all_vals, xSample); 
#             #     // }
#             #     // return all_vals[4]; // index 4 is the pressure 
#             # }
#             # """

#         else:
#             externalInteractionFunctions = ""

        # externalInteractionFunctions += """
        externalInteractionFunctions = """

void Model0d::initialise(double voiLoc, double *statesLoc, double *ratesLoc, double *varLoc,
                         const char* statesPath, const char* varsPath, bool useFirstRow)
{    
    // Default initialisation
    initialiseVariables(voiLoc, statesLoc, ratesLoc, varLoc);

    auto isValid = [](const char* p) { 
        if (p == nullptr) return false;

        std::string s(p);
        return (s != \"None\" && !s.empty()); 
    };

    // Load initial state from files and overwrite defaults
    if (isValid(statesPath)) {
        std::cout << \"Loading initial states from file: \" << statesPath << std::endl;
        loadFromFile(statesPath, statesLoc, STATE_INFO, STATE_COUNT, useFirstRow);
    }
    if (isValid(varsPath)) {
        std::cout << \"Loading initial variables from file: \" << varsPath << std::endl;
        loadFromFile(varsPath, varLoc, VARIABLE_INFO, VARIABLE_COUNT, useFirstRow);
    }
}

void Model0d::initialiseVariablesAndComputeConstants() {
    // initialiseVariables(voi, states, rates, variables);
    initialise(voi, states, rates, variables, initStatesPath, initVarsPath, false);
    
    computeComputedConstants(variables);
    getExternalVariablesVolume(variables);
    computeNonExternalVariables(voi, states, rates, variables);
    // in a coupled model, nothing has been received from the 1d solver yet: the received values are still 0 here.
    // These variables are only used in the t = 0 output row (initial conditions); the first evaluation of the rates
    // recomputes them after the first exchange.
    computeCouplingDependentVariables(voi, states, rates, variables);
    // computeRates(voi, dt, states, rates, variables); // is this needed here?
    // computeVariables(voi, dt, states, rates, variables); // variables all already computed within computeRates() function
        """
        if delay_variables:
            # delay variables: start the histories from the initial values of the variables to delay (computed above),
            # then recompute the variables that use the delayed values
            anchor = "    computeNonExternalVariables(voi, states, rates, variables);\n    // in a coupled model, nothing"
            assert externalInteractionFunctions.count(anchor) == 1
            externalInteractionFunctions = externalInteractionFunctions.replace(anchor,
                "    computeNonExternalVariables(voi, states, rates, variables);\n"
                "    initDelayHistories(); // delay variables: initial values of the variables to delay\n"
                "    computeNonExternalVariables(voi, states, rates, variables); // with the delayed values\n"
                "    // in a coupled model, nothing")
        if self.solver == 'CVODE':
            externalInteractionFunctions += """
    // reinitialise the CVODE solver with the initialised state variables (y wraps the states array without owning it:
    // the vector made in set_ode_solver() is destroyed first, otherwise it would leak)
    N_VDestroy_Serial(y);
    y = N_VMake_Serial(STATE_COUNT, states, context);
    CVodeReInit(solver, voi, y);
        """
        elif self.solver == 'PETSC':
            externalInteractionFunctions += """
    PetscScalar *y_arr;
    PETSC_CHECK(VecGetArray(y, &y_arr));
    std::copy(states, states + STATE_COUNT, y_arr);
    PETSC_CHECK(VecRestoreArray(y, &y_arr));

    PETSC_CHECK(TSSetSolution(ts, y)); // set the initial conditions
    PETSC_CHECK(TSSetTime(ts, voi)); // set the initial time
        """

        externalInteractionFunctions += """
}
        """ 

        
        # TODO make the integration scheme implementation general
        solveOneStepFunction = """
        """
#         if self.solver == 'CVODE':
#             solveOneStepFunction += """
# void Model0d::solveOneStepCVODE(double dtLoc)
# {
# """
#         else:
#             solveOneStepFunction += """
# void Model0d::solveOneStepExpl(double dtLoc)
# {
# """
        # Rounding of times to the grid of tolTime: a coupled model uses the function shared with the coupler
        # (and mirrored by the 1d solver), a fully 0d model (no coupler_mpi_common.h) the same formula.
        if self.couple_to_1d:
            solveOneStepFunction += """
double Model0d::roundTime(double t) const
{
    // same rounding as the coupler and the 1d solver (tolTime is COUPLING_TOL_TIME)
    return coupling_round_time(t);
}
"""
        else:
            solveOneStepFunction += """
double Model0d::roundTime(double t) const
{
    // round to the grid of tolTime, as coupling_round_time() in coupler_mpi_common.h
    const double invTolTime = std::round(1.0 / tolTime); // inverse of tolTime, as an integer
    return std::round(t * invTolTime) / invTolTime;
}
"""
        solveOneStepFunction += """
void Model0d::solveOneStep(double dtLoc)
{
"""
        if self.solver == 'CVODE' or self.solver == 'PETSC': 
            solveOneStepFunction += f"""
    if (solver_0d != \"{self.solver}\") {{
        std::cout << \"Model0d::solveOneStep :: ODE solver chosen is not {self.solver}: \" << solver_0d << std::endl;
        exit(1);
    }}
"""
        solveOneStepFunction += """    dt = dtLoc;
"""
        if self.couple_to_1d:
            solveOneStepFunction += """
    // matching global time step between 1d and 0d (at the start of each global time step: TAG_DT_STEP)
    if (N1d0dTot>0){
        zero_data_dt[0] = voi;
        zero_data_dt[1] = dt;
        MPI_Send(zero_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DT_STEP, MPI_COMM_WORLD);
    }

    if (N1d0dTot>0){
        MPI_Recv(parent_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DT_STEP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // continue from the agreed time and time step: the coupler has checked that the agreed time
        // matches voi within tolTime, so this only removes round-off differences with the 1d solver
        voi = parent_data_dt[0];
        dt = parent_data_dt[1];
    }
"""

        if self.solver == 'CVODE':
            solveOneStepFunction += """
    voiEnd = voi + dt;
    voiEnd = roundTime(voiEnd);

    states = N_VGetArrayPointer_Serial(y);

    CVodeSetStopTime(solver, voiEnd);
    int flag = CVode(solver, voiEnd, y, &voi, CV_NORMAL); 
    
    if (flag<0){ // error occurred
        std::cerr << \"CVODE :: Error occurred with flag \" << flag << \". Exiting.\" << std::endl;
        exit(1);
    }
    else if (flag==0){ // CV_SUCCESS
        // Successful function return.
    }
    else if (flag==1){ // CV_TSTOP_RETURN
        // CVode succeeded by reaching the specified stopping point.
        // std::cout << "CVODE :: Stop time reached with flag " << flag << std::endl;
    }
    else if (flag==2){ // CV_ROOT_RETURN
        // CVode succeeded and found one or more roots.
        std::cout << "CVODE :: Root found with flag " << flag << std::endl;
    }
    else if (flag==99){ // CV_WARNING
        // CVode succeeded but an unusual situation occurred.
        std::cout << "CVODE :: WARNING :: Unusual situation occurred with flag " << flag << ". CHECK YOUR OUTPUTS." << std::endl;
    }

    states = N_VGetArrayPointer_Serial(y);
"""
            if self.couple_to_1d:
                solveOneStepFunction += """
    if (N1d0dTot>0){
        // end of the global time step: the coupler checks the time reached and relays it to the 1d solver, which echoes it
        // (dt keeps the time step agreed at the start of the step, the one taken)
        zero_data_dt[0] = voi; // time reached at the end of the step, checked by the coupler
        zero_data_dt[1] = COUPLING_END_OF_STEP; // signal to coupler that time step is done
        MPI_Send(zero_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DT, MPI_COMM_WORLD);
        MPI_Recv(parent_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
"""
            solveOneStepFunction += """
    // computeVariables(voi, states, rates, variables);
    getExternalVariablesVolume(variables);
    getDelayVariables(voi, dt, states, variables);
    computeNonExternalVariables(voi, states, rates, variables);
    computeCouplingDependentVariables(voi, states, rates, variables); // with the values received at the last exchange
    storeDelayHistories(); // delay variables: values at the end of the time step
    userData->tOld = voi; // &voi is already incremented by dt within CVode() function
}
"""
        elif self.solver == 'PETSC':
            solveOneStepFunction += """
    voiEnd = voi + dt;
    // voiEnd = std::round(voiEnd / tolTime) * tolTime;

    // PETSC_CHECK: a failed call stops the program (see its definition)
    PETSC_CHECK(TSSetTime(ts, voi)); // start from voi (rounded, or the agreed time in a coupled run)
    PETSC_CHECK(TSSetTimeStep(ts, dt)); // set time step
    PETSC_CHECK(TSSetMaxTime(ts, voiEnd)); // set final time for the current one-step solve
    // Take one time step of size dt: TSSolve stops when it reaches the final time set above (after shorter steps if the
    // step had to be retried, see TSSetMaxSNESFailures in set_ode_solver()).
    // If the step fails, stop the program (as for CVODE). Returning instead would leave the time unchanged, and the
    // time loop would then repeat the same failing step forever.
    PetscErrorCode ierrSolve = TSSolve(ts, y);
    TSConvergedReason tsReason;
    PETSC_CHECK(TSGetConvergedReason(ts, &tsReason));
    if (ierrSolve != PETSC_SUCCESS || tsReason < 0) {
        std::cerr << "0d solver :: ERROR: PETSc TSSolve failed in the time step from " << voi << " to " << voiEnd
                  << " (" << TSConvergedReasons[tsReason] << "). Exiting." << std::endl;
        exit(1);
    }

    PetscReal t;
    PETSC_CHECK(TSGetTime(ts, &t));
    // TSSolve can also stop before the final time without failing, e.g. when the TS object reaches its maximum number
    // of steps: the time would then not advance, and the time loop would repeat this step forever
    if (std::fabs(t - voiEnd) > tolTime) {
        std::cerr << "0d solver :: ERROR: PETSc TSSolve stopped at t = " << t << " before the end of the time step from "
                  << voi << " to " << voiEnd << " (" << TSConvergedReasons[tsReason] << "). Exiting." << std::endl;
        exit(1);
    }
    voi = roundTime(t); // rounded as in the other solvers, so that it cannot drift

    const PetscScalar *y_arr;
    PETSC_CHECK(VecGetArrayRead(y, &y_arr));
    std::copy(y_arr, y_arr + STATE_COUNT, states);
    PETSC_CHECK(VecRestoreArrayRead(y, &y_arr));

"""

            if self.couple_to_1d:
                solveOneStepFunction += """
    if (N1d0dTot>0){
        // end of the global time step: the coupler checks the time reached and relays it to the 1d solver, which echoes it
        // (dt keeps the time step agreed at the start of the step, the one taken)
        zero_data_dt[0] = voi; // time reached at the end of the step, checked by the coupler
        zero_data_dt[1] = COUPLING_END_OF_STEP; // signal to coupler that time step is done
        MPI_Send(zero_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DT, MPI_COMM_WORLD);
        MPI_Recv(parent_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
"""
            solveOneStepFunction += """
    // computeVariables(voiEnd, states, rates, variables);
    getExternalVariablesVolume(variables);
    getDelayVariables(voi, dt, states, variables);
    computeNonExternalVariables(voi, states, rates, variables);
    computeCouplingDependentVariables(voi, states, rates, variables); // with the values received at the last exchange
    storeDelayHistories(); // delay variables: values at the end of the time step
    userData->tOld = voi;
}
"""

        else:
            solveOneStepFunction += """
    if (solver_0d == \"explEul\") {
        // Step 1
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states0[i] = states[i];
        }
        computeRates(voi, 0.0*dt, states0, k1, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            rates[i] = k1[i];
            states[i] = states0[i] + dt*rates[i];
        }
        // //computeVariables(voi+dt, states, rates, variables);
        // getExternalVariablesVolume(variables);
        // computeNonExternalVariables(voi+dt, states, rates, variables);
        // getDelayVariables(voi, dt, states, variables);
    }
    else if (solver_0d == \"Heun\") {
        // Step 1
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states0[i] = states[i];
        }
        computeRates(voi, 0.0*dt, states0, k1, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + dt*k1[i];
        }
        // computeNonExternalVariables(voi+dt, states, k1, variables);

        // Step 2
        computeRates(voi+dt, dt, states, k2, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            rates[i] = 0.5*(k1[i] + k2[i]);
            states[i] = states0[i] + dt*rates[i];
        }
        // // computeVariables(voi+dt, states, rates, variables);
        // getExternalVariablesVolume(variables);
        // computeNonExternalVariables(voi+dt, states, rates, variables);
        // getDelayVariables(voi, dt, states, variables);
    }
    else if (solver_0d == \"midpoint\") {
        // Step 1
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states0[i] = states[i];
        }
        computeRates(voi, 0.0*dt, states0, k1, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + 0.5*dt*k1[i];
        }
        // computeNonExternalVariables(voi+0.5*dt, states, k1, variables);

        // Step 2
        computeRates(voi+0.5*dt, 0.5*dt, states, k2, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + dt*k2[i];
        }
        // // computeVariables(voi+dt, states, k2, variables);
        // getExternalVariablesVolume(variables);
        // computeNonExternalVariables(voi+dt, states, k2, variables);
        // getDelayVariables(voi, dt, states, variables);
    }
    else if (solver_0d == \"RK4\") {
        // Step 1
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states0[i] = states[i];
        }
        computeRates(voi, 0.0*dt, states0, k1, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + 0.5*dt*k1[i];
        }
        // computeNonExternalVariables(voi+0.5*dt, states, k1, variables);
        
        // Step 2
        computeRates(voi+0.5*dt, 0.5*dt, states, k2, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + 0.5*dt*k2[i];
        }
        // computeNonExternalVariables(voi+0.5*dt, states, k2, variables);
    
        // Step 3
        computeRates(voi+0.5*dt, 0.5*dt, states, k3, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            states[i] = states0[i] + dt*k3[i];
        }
        // computeNonExternalVariables(voi+dt, states, k3, variables);
        
        // Step 4
        computeRates(voi+dt, dt, states, k4, variables);
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            rates[i] = wRK4[0]*k1[i] + wRK4[1]*k2[i] + wRK4[2]*k3[i] + wRK4[3]*k4[i];
            states[i] = states0[i] + dt*rates[i];
        }
        // // computeVariables(voi+dt, states, rates, variables);
        // getExternalVariablesVolume(variables);
        // computeNonExternalVariables(voi+dt, states, rates, variables);
        // getDelayVariables(voi, dt, states, variables);   
    }
    else{
        std::cout << \"Model0d::solveOneStep :: explicit ODE solver not defined \" << solver_0d << std::endl;
        exit(1);
    }
"""
            # explicit schemes: stop at the first state that is not finite after the step (CVODE and PETSc detect a
            # failing solution themselves); in a coupled model, the values received from the 1d solver are a possible
            # cause too
            received_hint = (" << \" (or check the values received from the 1d solver)\"" if self.couple_to_1d else "")
            solveOneStepFunction += """
    // Stop if the step gave a state that is not finite (inf or NaN), e.g. with a time step too large for the fastest
    // dynamics of the model, instead of continuing and writing NaN results
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        if (!std::isfinite(states[i])) {
            std::cerr << \"0d solver :: ERROR: the \" << solver_0d << \" solution is not finite at t = \" << roundTime(voi+dt)
                      << \" s: state \" << STATE_INFO[i].component << \"/\" << STATE_INFO[i].name << \" = \" << states[i]
                      << \". The solver time step (\" << dt << \" s) is probably too large for this model: use a smaller \"
                      << \"MaximumStep in solver_info, or the CVODE or PETSc solver\"""" + received_hint + """ << std::endl;
            exit(1);
        }
    }

    voi += dt;
    voi = roundTime(voi);
"""

            if self.couple_to_1d:
                solveOneStepFunction += """
    if (N1d0dTot>0){
        // end of the global time step: the coupler checks the time reached and relays it to the 1d solver, which echoes it
        // (dt keeps the time step agreed at the start of the step, the one taken)
        zero_data_dt[0] = voi; // time reached at the end of the step, checked by the coupler
        zero_data_dt[1] = COUPLING_END_OF_STEP; // signal to coupler that time step is done
        MPI_Send(zero_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DT, MPI_COMM_WORLD);
        MPI_Recv(parent_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
"""

            solveOneStepFunction += """
    getExternalVariablesVolume(variables);
    getDelayVariables(voi, dt, states, variables);
    computeNonExternalVariables(voi, states, rates, variables);
    computeCouplingDependentVariables(voi, states, rates, variables); // with the values received at the last exchange
    storeDelayHistories(); // delay variables: values at the end of the time step
}
"""


#         if self.solver == 'forward_euler':
#             solveOneStepFunction += """
            
#     time_dof_0d = 0;
#     computeRates(voi, states, rates, variables);

#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         // simple forward Euler integration
#         states[i] = states[i] + dt * rates[i];
#     time_dof_0d = -1;
#     computeVariables(voi, states, rates, variables);
#     voi += dt;
#     }
# }
#             """
#         elif self.solver == 'RK4':
#             solveOneStepFunction += """
#     // RK4 integration
#     // first step: calculate k1
#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         temp_states[i] = states[i];
#     }
#     time_dof_0d = 0;
#     computeRates(voi, temp_states, k1, variables);

#     // second step: calculate k2
#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         temp_states[i] = states[i] + dt/2.0 * k1[i];
#     }
#     time_dof_0d = 1;
#     computeRates(voi+dt/2.0, temp_states, k2, variables);
    
#     // third step: calculate k3
#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         temp_states[i] = states[i] + dt/2.0 * k2[i];
#     }
#     time_dof_0d = 2;
#     computeRates(voi+dt/2.0, temp_states, k3, variables);
    
#     // third step: calculate k4
#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         temp_states[i] = states[i] + dt * k3[i];
#     }
#     time_dof_0d = 3;
#     computeRates(voi+dt, temp_states, k4, variables);

#     for (size_t i = 0; i < STATE_COUNT; ++i) {
#         rates[i] = 1.0/6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
#         states[i] = states[i] + dt * rates[i];
#     }
#     // when we set time_dof_0d to -1, we only get q_BC
#     // from the ones evaluated at the 4 time dofs. we dont step the fluct
#     time_dof_0d = -1;
#     computeVariables(voi, states, rates, variables);
#     voi += dt;
#         """
#             if self.couple_to_1d:
#                 solveOneStepFunction += """
#         // TODO Finbar, is the below correct? it was used in the 1d code.
#     model1d_ptr->correctTimeLTS(voi,model1d_ptr->dtMaxLTS);
# }
#         """
#             else:
#                 solveOneStepFunction += """
# }
#         """

#         elif self.solver == 'CVODE':
#             solveOneStepFunction += """ 

#     voiEnd = voi + dt;
#     // CVodeSetStopTime(solver, voiEnd);

#     CVode(solver, voiEnd, y, &voi, CV_NORMAL);

#     // Compute our variables.

#     states = N_VGetArrayPointer_Serial(y);
#     computeVariables(voiEnd, states, rates, variables);
#     voi += dt;
#     }
#             """
        

        # print("#################################################")
        # print("solveOneStepFunction")
        # print(solveOneStepFunction)
        

        # typedef void (*computeRatesType)(double, double *, double *, double *);
        if self.solver == 'CVODE':
            userDataHeader = """
class UserOdeData
{
public:

    explicit UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates, void* pCvodeMem);

    double* variables() const;
    Model0d::FunctionType computeRates() const;
    void* cvodeMem() const;

    sunrealtype tOld = 0.0;
    long int nstOld = 0;

private:
    double *mVariables;
    Model0d::FunctionType mComputeRates;
    void *mCvodeMem;
};

"""
        else:
            userDataHeader = """
class UserOdeData
{
public:

    explicit UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates);

    double* variables() const;
    Model0d::FunctionType computeRates() const;
    double tOld = 0.0;

private:
    double *mVariables;
    Model0d::FunctionType mComputeRates;
};

"""
        UserDataCC = """
        """

        if self.solver == 'CVODE':
            UserDataCC += """
UserOdeData::UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates, void* pCvodeMem) :
    mVariables(pVariables),
    mComputeRates(pComputeRates),
    mCvodeMem(pCvodeMem)
{
}
"""
        else:
            UserDataCC += """
UserOdeData::UserOdeData(double *pVariables, Model0d::FunctionType pComputeRates) :
    mVariables(pVariables),
    mComputeRates(pComputeRates)
{
}
"""

        UserDataCC += """
//==============================================================================

double * UserOdeData::variables() const
{
    // Return our algebraic array
    return mVariables;
}

//==============================================================================

Model0d::FunctionType UserOdeData::computeRates() const
{
    // Return our compute rates function
    return mComputeRates;
}
"""
        if self.solver == 'CVODE':
            UserDataCC += """
//==============================================================================

void* UserOdeData::cvodeMem() const
{
    return mCvodeMem;
}
"""


        if self.solver == 'CVODE':
            funcCC = f"""
int func(double voi_, N_Vector y, N_Vector yDot, void *userData)
{{
    // Function that computes the RHS of the ODE system, in the form of f(t, y, ydot, user_data)
    // - t : the current value of the independent variable (time)
    // - y : the current value of the dependent variable vector (the states)
    // - ydot : the output vector f(t,y) (the rates)
    // - user_data : the user_data pointer

    UserOdeData *realUserData = static_cast<UserOdeData*>(userData);
    sunrealtype t_solver;
    sunrealtype hcur;
    long int nst;
    sunrealtype dtLoc;

    CVodeGetCurrentTime(realUserData->cvodeMem(), &t_solver);
    CVodeGetCurrentStep(realUserData->cvodeMem(), &hcur); // current internal time step
    CVodeGetNumSteps(realUserData->cvodeMem(), &nst); // total internal steps so far

    if (nst == realUserData->nstOld){{
        // CVODE did not take a step
        dtLoc = 0.0;
        // std::cout << std::setprecision(20)
        //             << "WARNING: CVODE did not take a step at time " << voi_
        //             << " with solver time " << t_solver
        //             << std::endl;
    }} else {{
        dtLoc = hcur;
        realUserData->nstOld = nst;
    }} 
    // std::cout << std::setprecision(20)
    //                 << "CVODE has internal time step " << dtLoc
    //                 << " at time " << voi_
    //                 << " with previous time " << realUserData->tOld
    //                 << std::endl;

    realUserData->tOld = voi_;

    realUserData->computeRates()(voi_, dtLoc, N_VGetArrayPointer_Serial(y), 
                            N_VGetArrayPointer_Serial(yDot), realUserData->variables());
    
    return 0;
}}
           
        """
            
        elif self.solver == 'PETSC':
            funcCC = """
// Convergence test of the Newton iterations of the implicit PETSc methods, as in CVODE: converged when the last
// Newton update is small compared with the tolerances of the solution (TSSetTolerances), i.e. when its weighted RMS
// norm, with weights 1/(reltol*|y_i| + abstol), is below 0.1. PETSc's default tests are kept too. They alone are not
// enough: they compare the residual with its initial value, and in stiff models (e.g. terminals with a small inertance)
// the initial guess of BDF can already be as good as round-off allows, so that the residual cannot decrease further
// and the Newton iterations fail.
PetscErrorCode SNESConvergedWRMS(SNES snes, PetscInt it, PetscReal xnorm, PetscReal snorm, PetscReal fnorm,
                                 SNESConvergedReason *reason, void *ctx)
{
    PetscFunctionBeginUser;
    // PETSc default tests
    PetscCall(SNESConvergedDefault(snes, it, xnorm, snorm, fnorm, reason, NULL));
    if (*reason != SNES_CONVERGED_ITERATING || it == 0) PetscFunctionReturn(PETSC_SUCCESS);

    // New CVODE-style convergence test:
    // accepts once the last Newton update is below 0.1 of the solution tolerances
    TS ts = static_cast<TS>(ctx);
    PetscReal abstol, reltol;
    PetscCall(TSGetTolerances(ts, &abstol, NULL, &reltol, NULL));
    Vec x, dx;
    PetscCall(SNESGetSolution(snes, &x));
    PetscCall(SNESGetSolutionUpdate(snes, &dx)); // last Newton update (full step: basic line search)
    const PetscScalar *x_arr, *dx_arr;
    PetscInt n;
    PetscCall(VecGetLocalSize(x, &n));
    PetscCall(VecGetArrayRead(x, &x_arr));
    PetscCall(VecGetArrayRead(dx, &dx_arr));
    PetscReal sum = 0.0;
    for (PetscInt i = 0; i < n; ++i) {
        const PetscReal w = reltol*PetscAbsScalar(x_arr[i]) + abstol;
        sum += PetscRealPart((dx_arr[i]/w)*(dx_arr[i]/w));
    }
    PetscCall(VecRestoreArrayRead(x, &x_arr));
    PetscCall(VecRestoreArrayRead(dx, &dx_arr));
    if (n > 0 && PetscSqrtReal(sum/n) < 0.1) *reason = SNES_CONVERGED_USER;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// PETSc RHS callable function
PetscErrorCode TSRHSfunc(TS ts, PetscReal t, Vec Y, Vec Ydot, void *ctx)
{
    PetscFunctionBeginUser;
    // Function that computes the RHS of the ODE system dy/dt.
    // void *ctx : generic pointer that PETSc passes straight through this RHS function, to let you attach any data you want to the solver.

    // Recover the C++ object that owns the model data
    Model0d *model = static_cast<Model0d*>(ctx); 

    // Temporary pointers that will point into the PETSc vectors
    const PetscScalar *y_ptr; // read-only view of the current state vector
    PetscScalar *ydot_ptr; // writable view where we will store dy/dt
    
    // Get the local time step that the solver is using right now.
    PetscReal dtLoc = model->dt;
    
    // PetscReal tEndLoc;
    // TSGetMaxTime(ts, &tEndLoc); 
    // std::cout << std::setprecision(16)
    //             << "TSRHSfunc :: called at time " << t
    //             << " || dt = " << dtLoc
    //             << " || final time = " << tEndLoc
    //             << " || old time = " << model->userData->tOld
    //             << std::endl;
    
    // Get raw C-style arrays from the PETSc vectors (PetscCall: an error is returned to TSSolve, which then fails).
    PetscCall(VecGetArrayRead(Y, &y_ptr)); // read-only access to Y  (the state vector)
    PetscCall(VecGetArray(Ydot, &ydot_ptr)); // read/write access to Ydot (the RHS vector)

    // Call the computeRates routine for computing the RHS, which expects plain double* pointers.
    model->computeRates(t, dtLoc, 
                        (double*)y_ptr, 
                        (double*)ydot_ptr, 
                        model->variables);

    // Restore the PETSc vectors / Release the raw arrays.
    PetscCall(VecRestoreArrayRead(Y, &y_ptr));
    PetscCall(VecRestoreArray(Ydot, &ydot_ptr));
    
    PetscFunctionReturn(PETSC_SUCCESS);
}

        """


        delayHistoryHeader = """
#include <deque>
#include <utility>
#include <algorithm>

// History of a variable to delay: its values at the end of the time steps (of any length), to compute its delayed
// value by linear interpolation (see Model0d::getDelayVariables()).
class DelayHistory {
public:
    void reset(double t, double value);   // a single value, also used before t (initial value of the variable)
    void add(double t, double value);     // value at the end of a time step
    double valueAt(double t) const;       // linear interpolation (first or last value outside the history)
    void dropBefore(double t);            // drop the values before t that are no longer needed
private:
    std::deque<std::pair<double, double>> points; // (time, value), with increasing times
};
        """

        delayHistoryCode = """
// History of a variable to delay (see the class in the header)
void DelayHistory::reset(double t, double value)
{
    points.clear();
    points.emplace_back(t, value);
}

void DelayHistory::add(double t, double value)
{
    // a time step ending at or before the last time stored replaces the values from that time on
    while (!points.empty() && points.back().first >= t) points.pop_back();
    points.emplace_back(t, value);
}

double DelayHistory::valueAt(double t) const
{
    if (t <= points.front().first) return points.front().second;
    if (t >= points.back().first) return points.back().second;
    auto after = std::upper_bound(points.begin(), points.end(), t,
                                  [](double tt, const std::pair<double, double>& p) { return tt < p.first; });
    auto before = after - 1;
    return before->second + (t - before->first)/(after->first - before->first)*(after->second - before->second);
}

void DelayHistory::dropBefore(double t)
{
    // keep the last value at or before t, needed to interpolate at t
    while (points.size() > 1 && points[1].first <= t) points.pop_front();
}

        """


        #XXX MAIN FUNCTION
        # When the 0d solver is coupled with a Pyhton/C++ model that does the simulation 
        # (via the other solver or a coupler), the main() function  below is not needed.
        mainScript = """
int main(int argc, char* argv[]){
"""
        if self.solver == 'PETSC':
            mainScript += """
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    if (ierr != PETSC_SUCCESS) {
        std::cerr << "0d solver :: ERROR: PetscInitialize failed with error code " << ierr << ". Exiting." << std::endl;
        return 1;
    }
    std::cout << "PETSc :: initialization complete " << ierr << std::endl;

    std::cout << "Starting 0D model simulation..." << std::endl;
    {
"""
        else:
            mainScript += """
    std::cout << "Starting 0D model simulation..." << std::endl;
"""

        mainScript += f"""
    // Time window of the user inputs (yaml file), as for the other model types: the model runs from 0 to
    // pre_time + sim_time, and its results are saved every dtSample (dt of the user inputs) from pre_time.
    const double pre_time = {self.pre_time!r};
    const double sim_time = {self.sim_time!r};
    const double dtSample = {self.dtSample!r}; // output time step
    double dt = {self.dtSolver}; // solver time step
    std::string ODEsolver = \"{self.solver}\";

    Model0d model0d_inst;
    model0d_inst.set_ode_solver(ODEsolver);
    std::string resFold = \"{self.results_dir}/\"; // results folder, set at generation (cpp_results_dir of the user inputs)
    model0d_inst.openOutputFiles(resFold);
    model0d_inst.voi = 0.0;
    model0d_inst.dt = dt;
    model0d_inst.initialiseVariablesAndComputeConstants();
    """
        mainScript += """
    // all the times are rounded to the grid of tolTime, as the time of the model (see roundTime())
    const double tolTime = model0d_inst.tolTime;
    const double save_time = model0d_inst.roundTime(pre_time);
    const double end_time = model0d_inst.roundTime(pre_time + sim_time);
    double next_output_time = model0d_inst.roundTime(dtSample);
    std::cout << "Simulating from 0 to " << end_time << " s, saving the results every " << dtSample
              << " s from " << save_time << " s" << std::endl;

    if (model0d_inst.voi >= save_time-tolTime){
        model0d_inst.writeOutput(model0d_inst.voi);
    }

    while (model0d_inst.voi < end_time-tolTime) {
        // solver time step, shortened to reach the next output time exactly
        double dtLoc = dt;
        if (model0d_inst.voi+dtLoc > next_output_time) {
            dtLoc = next_output_time-model0d_inst.voi;
        }
        model0d_inst.solveOneStep(dtLoc);

        if (model0d_inst.voi >= next_output_time-tolTime) {
            next_output_time = model0d_inst.roundTime(next_output_time+dtSample);
            if (model0d_inst.voi >= save_time-tolTime){
                model0d_inst.writeOutput(model0d_inst.voi);
            }
        }
    }
    model0d_inst.closeOutputFiles();

    // TODO autogenerate a dict of variable names to print with variables

    printf("################################################################################\\n");
    printf("Final values:\\n");
    printf("  time: ");
    printf("%f\\n", model0d_inst.voi);
    printf("  states:\\n");
    for (size_t i = 0; i < model0d_inst.STATE_COUNT; ++i) {
        printf("%f\\n", model0d_inst.states[i]);
    }
    printf("  variables:\\n");
    for (size_t i = 0; i < model0d_inst.VARIABLE_COUNT; ++i) {
        printf("%f\\n", model0d_inst.variables[i]);
    }
    printf("################################################################################\\n");
"""
        if self.solver == 'PETSC':
            mainScript += """
    }

    ierr = PetscFinalize();
    std::cout << "PETSc :: finalization complete " << ierr << std::endl;

    printf("Done.\\n");
    return 0;
}
"""
        else:
            mainScript += """
    printf("Done.\\n");
    return 0;
}
"""

        # save header to file
        with open(os.path.join(self.cpp_generated_models_dir, f'{self.output_cpp_file_name}.h'), 'w') as f:

            f.write(preHeaderStuff) 
            if delay_variables:
                f.write(delayHistoryHeader)
            f.write(interFaceCodePreClass)
            f.write(classInitHeader)
            f.write(interFaceCodeInClass)
            # f.write(externalInteractionHeaders)
            f.write(otherHeaderInits)
            if delay_variables:
                f.write(delayHistoriesMember)
            f.write(classFinisherHeader)
            f.write(userDataHeader)
   
        # save implementation to file
        with open(os.path.join(self.cpp_generated_models_dir, f'{self.output_cpp_file_name}.cc'), 'w') as f:

            f.write(preClassStuff)
            f.write(preSourceStuff) # this has to be below the #include "model.h" in implementationCode()
            if self.solver == 'CVODE' or self.solver == 'PETSC':
                f.write(funcCC)
            f.write(classInit)
            f.write(postClassInit)
            # f.write(non_external_variables_function)
            f.write(externalInteractionFunctions)

            f.write(solverInitFunction) #XXX Bea's addition
            f.write(couplingFunctions) #XXX Bea's addition
            f.write(outputFunctions) #XXX Bea's addition

            f.write(computeEV)
            f.write(solveOneStepFunction)
            f.write(UserDataCC)
            
            if delay_variables:
                f.write(delayHistoryCode)
            
            # if self.create_main:
            if (self.create_main and not self.couple_to_1d):
                f.write(mainScript)
                os.makedirs(self.results_dir, exist_ok=True) # results folder written in the main (see __init__)
        mark_parameters_maybe_unused(os.path.join(self.cpp_generated_models_dir, f'{self.output_cpp_file_name}.cc'))
        
        
        #XXX MAIN SCRIPT FILE (for coupled models only)
        if (self.create_main and self.couple_to_1d):
            #XXX create also the main script main0d.cpp to be used to run the 0d model from the 1d-0d coupler
            with open(generators_dir_path+'/main0dTemplate.cpp', 'r') as file:
                lines = file.readlines()

            mainScript = """
"""

            # main0d only runs coupled: it gets the ODE solver, T0, pre_time, sim_time and the model name from the
            # coupler configuration file (passed by launch_coupled_mpi.bash)
            for line in lines:
                if self.solver == 'PETSC':
                    if '// #include <petscsys.h>' in line:
                        line = line.replace('// #include <petscsys.h>', '#include <petscsys.h>')
                    if 'PetscErrorCode' in line:
                        line = line.replace('// ', '')
                    if 'PetscInitialize' in line:
                        line = line.replace('// ', '')
                    if 'PetscFinalize' in line:
                        line = line.replace('// ', '')
                    if 'PETSC_COMM_WORLD' in line: # PETSc only on the 0d solver communicator, not the whole MPI job
                        line = line.replace('// ', '')

                if 'std::string resFold =' in line:
                    # results in <results_dir>/<file_prefix>/ (see __init__), as for a fully 0d model
                    os.makedirs(self.results_dir, exist_ok=True)
                    results0d_folder_tmp = "./"
                    line = line.replace(f"std::string resFold = \"{results0d_folder_tmp}\";", f"std::string resFold = \"{self.results_dir}/\";")
                
                if 'const double dtSample = ' in line:
                    idx0 = line.find("=")
                    idx1 = line.find(";")
                    line = line.replace(line[idx0:idx1+1], f"= {self.dtSample};")

                if 'const double dt0D = ' in line:
                    idx0 = line.find("=")
                    idx1 = line.find(";")
                    line = line.replace(line[idx0:idx1+1], f"= {self.dtSolver};")

                # if 'model0d->solveOneStep(' in line:
                #     if self.solver == 'CVODE':
                #         line = line.replace("solveOneStep(", "solveOneStepCVODE(")
                #     else:
                #         line = line.replace("solveOneStep(", "solveOneStepExpl(")
                
                mainScript += line # + '\n'

            # save main to file
            with open(os.path.join(self.cpp_generated_models_dir, 'main0d.cpp'), 'w') as f:
                f.write(mainScript)


        #XXX MAKEFILE and BASH SCRIPT to build and run the 0d solver (or the 1d-0d coupler)
        if not self.couple_to_1d:
            if self.solver == 'PETSC':
                makefile = "MakefilePETSC"
                runfile = "runPETSC.bash"
            else:
                makefile = "Makefile"
                runfile = "runCVODE.bash"
            
            # Makefile for 0d solver
            with open(os.path.join(generators_dir_path, '../makefiles_cpp/cpp_0d/'+makefile), 'r') as file:
                lines = file.readlines()

            makefileScript = """
"""
            for line in lines:
                if 'model0d' in line:
                    line = line.replace('model0d', self.output_cpp_file_name)
                makefileScript += line

            with open(os.path.join(self.cpp_generated_models_dir, makefile), 'w') as f:
                f.write(makefileScript)

            # bash script to build and run the 0d solver
            source = os.path.join(generators_dir_path, '../makefiles_cpp/cpp_0d/'+runfile)
            with open(source, 'r') as file:
                lines = file.readlines()

            runfileScript = "" # no empty line before the first line '#!/bin/bash'
            for line in lines:
                if 'model0d' in line:
                    line = line.replace('model0d', self.output_cpp_file_name)
                runfileScript += line

            destination = os.path.join(self.cpp_generated_models_dir, runfile)
            with open(destination, 'w') as f:
                f.write(runfileScript)
            shutil.copymode(source, destination) # executable, as the template
        
        else:
            if self.solver == 'PETSC':
                makefile = "MakefilePETSC"
            else:
                makefile = "Makefile"
            # makefile_coupler = "MakefileCoupler"
            runfile_coupler = "run_coupler1d0d.bash"
            
            # Makefile for 0d solver         
            source = os.path.join(generators_dir_path, '../makefiles_cpp/cpp_hybrid/'+makefile)
            destination = os.path.join(self.cpp_generated_models_dir, makefile) 
            shutil.copy2(source, destination)

            # header shared by the coupler and the 0d solver for the MPI coupling (included by main0d.cpp and model0d.cc)
            source = os.path.join(generators_dir_path, '../coupler/coupler_mpi_common.h')
            destination = os.path.join(self.cpp_generated_models_dir, 'coupler_mpi_common.h')
            shutil.copy2(source, destination)

            # Makefile for coupler
            # source = os.path.join(generators_dir_path, '../makefiles_cpp/cpp_hybrid/'+makefile_coupler)
            # destination = os.path.join(self.cpp_generated_models_dir, makefile_coupler) 
            # shutil.copy2(source, destination)

            # bash script to build and run the 1d-0d coupled solver
            source = os.path.join(generators_dir_path, '../makefiles_cpp/cpp_hybrid/'+runfile_coupler)
            with open(source, 'r') as file:
                lines = file.readlines()

            runfileScript = "" # no empty line before the first line '#!/bin/bash'
            for line in lines:
                if ('USE_PETSC' in line and self.solver == 'PETSC'):
                    line = line.replace('USE_PETSC=0', 'USE_PETSC=1')
                    # print(line)
                if 'FOLDERcpp=\"\"' in line:
                    line = line.replace('FOLDERcpp=\"\"', f'FOLDERcpp=\"{self.cpp_generated_models_dir}/\"')
                    # print(line)
                if 'FOLDERcoupler=\"\"' in line:
                    line = line.replace('FOLDERcoupler=\"\"', f'FOLDERcoupler=\"{generators_dir_path}/../coupler/\"')
                    # print(line)
                runfileScript += line

            destination = os.path.join(self.cpp_generated_models_dir, runfile_coupler)
            with open(destination, 'w') as f:
                f.write(runfileScript)
            shutil.copymode(source, destination) # executable, as the template

            # coupler configuration file
            with open(os.path.join(generators_dir_path, '../makefiles_cpp/cpp_hybrid/coupler_configTemplate.json'), 'r') as file:
                coupler_config = json.load(file)

            coupler_config['inputFold'] = self.cpp_generated_models_dir+'/'
            coupler_config['networkName'] = self.file_prefix
            coupler_config['ODEsolver'] = self.solver
            # PETSc time integration scheme of both solvers, checked by the coupler (see CouplingPetscScheme in
            # coupler_mpi_common.h): only with the PETSc ODE solver
            if self.solver == 'PETSC':
                coupler_config['PETSc_solver'] = self.petsc_solver
            else:
                coupler_config.pop('PETSc_solver', None)
            # time settings: cardiac period (parameter T of the model) and time window (pre_time and sim_time of the
            # user inputs), used by the coupler and both solvers (see coupling_end_time() in coupler_mpi_common.h)
            coupler_config['T0'] = self.T0
            coupler_config['pre_time'] = self.pre_time
            coupler_config['sim_time'] = self.sim_time
            coupler_config['solver0d_path'] = self.cpp_generated_models_dir+'/main0d'
            # coupler_config['python_path'] = '/home/bghi639/anaconda3/bin/python'
            # coupler_config['python_path'] = '/home/bghi639/Software/OpenCOR-0-8-3-Linux/python'
            coupler_config['python_path'] = opencor_root+'/python' #XXX DEFAULT OpenCOR Python
            # Python 1d solver (run1DBFmodel.py, in its own repository): solver_1d_path of the user inputs
            coupler_config['solver1d_path'] = self.solver_1d_path if self.solver_1d_path else ''
            coupler_config['initFile_sim1d_path'] = self.model_1d_config_path
            coupler_config['initStatePath'] = 'None'

            with open(os.path.join(self.cpp_generated_models_dir, 'coupler_config.json'), 'w') as f:
                json.dump(coupler_config, f, indent=4)

        
        print("Cpp files generated. Check they run properly.")
        
        return True # TODO check if they run properly


#XXX Bea: GENERATION OF 1D INPUT FILES (for my 1D Python solver)
class CVS1DPythonGenerator(object):
    '''
    Generates Python files for 1D model.
    '''

    def __init__(self, model, file_prefix, vessels1d_csv_abs_path, parameters_csv_abs_path,
                model_1d_config_path, generated_model_subdir, cpp_generated_models_dir=None,
                solver='CVODE', dtSample=1e-3, dtSolver=1e-4, pre_time=None, sim_time=None, T0=None,
                petsc_solver=None, conn_1d_0d_info=None):
        '''
        Constructor
        pre_time, sim_time: time window of the user inputs (see get_time_window())
        T0: cardiac period, parameter T of the model (see get_cardiac_period())
        petsc_solver: PETSc time integration scheme, the same as in the 0d solver (see get_petsc_solver())
        '''

        self.model = model
        self.file_prefix = file_prefix
        # absolute path: its folder is written into input.ini (testFold), which the 1d solver reads from another folder
        self.initFile1d = os.path.abspath(model_1d_config_path)
        
        self.run1dFold = os.path.dirname(self.initFile1d)
        self.initFiles1dFold = self.run1dFold+'/input_files'
        # creates the 1d run folder and its input_files folder, with all their missing parent folders
        # (e.g. files_1d/<model>/run000/input_files when files_1d does not exist yet)
        os.makedirs(self.initFiles1dFold, exist_ok=True)

        # print(self.file_prefix)
        # print(self.initFile1d)
        # print(self.run1dFold, os.path.exists(self.run1dFold))
        # print(self.initFiles1dFold, os.path.exists(self.initFiles1dFold))

        self.ODEsolver = solver
        self.petsc_solver = get_petsc_solver(petsc_solver, True)
        self.dtSample = dtSample
        self.dtSolver = dtSolver
        self.pre_time, self.sim_time, self.dtSample = get_time_window(pre_time, sim_time, dtSample)
        if not (T0 is not None and np.isfinite(T0) and T0 > 0.0):
            print(f"CVS1DPythonGenerator __init__ :: ERROR: the 1d solver needs the cardiac period T0 > 0 "
                  f"(parameter T, see get_cardiac_period()), not {T0}")
            exit(1)
        self.T0 = T0
        self.conn_1d_0d_info = conn_1d_0d_info

        self.generated_model_subdir = generated_model_subdir
        if cpp_generated_models_dir is None:
            self.cpp_generated_models_dir = self.generated_model_subdir + "_cpp"
        else:
            self.cpp_generated_models_dir = cpp_generated_models_dir
        self.cpp_generated_models_dir = os.path.abspath(self.cpp_generated_models_dir) # as in CVS0DCppGenerator

        self.csv_parser = CSVFileParser()
        self.vessels_df = self.csv_parser.get_data_as_dataframe_multistrings(vessels1d_csv_abs_path, True) 
        self.params_df = self.csv_parser.get_data_as_dataframe_multistrings(parameters_csv_abs_path, True)

        self.vessFileName = self.initFiles1dFold+f'/vess_{self.file_prefix[:-3]}.txt'
        self.nodeFileName = self.initFiles1dFold+f'/nodes_{self.file_prefix[:-3]}.txt'
        self.nameFileName = self.initFiles1dFold+f'/names_{self.file_prefix[:-3]}.csv'


    def generate_files(self):
        print("Generating 1D Python files...")

        # 1: here we need to generate  
        # - vess file in self.initFiles1dFold
        # - nodes file in self.initFiles1dFold
        # - names file in self.initFiles1dFold
        vess1d, nodes1d = generate1DPythonModelFiles(self.vessels_df, self.params_df, self.vessFileName, self.nodeFileName, self.nameFileName, self.conn_1d_0d_info)

        # 2: update the input.ini file in self.run1dFold
        computeTotBV = False
        for i in range(len(self.conn_1d_0d_info)):
            if "port_volume_sum" in self.conn_1d_0d_info[str(i+1)]:
                if self.conn_1d_0d_info[str(i+1)]["port_volume_sum"]==1:
                    computeTotBV = True
        generate1DPythonSimInitFile(self.params_df, vess1d, nodes1d, self.initFile1d, self.file_prefix, self.run1dFold, self.ODEsolver, self.dtSample, computeTotBV,
                                    self.T0, self.pre_time, self.sim_time, petsc_solver=self.petsc_solver)


        if self.file_prefix.endswith("_1d"):
            json_filename = self.file_prefix[:-3]+"_coupler1d0d.json"
        else:
            json_filename = self.file_prefix+"_coupler1d0d.json"
        with open(self.initFiles1dFold+"/"+json_filename, "w") as f:
            # vess1d_name is used only by the generators (the coupler reads this file as integer values only)
            json.dump({k: {kk: vv for kk, vv in conn.items() if kk != "vess1d_name"}
                       for k, conn in self.conn_1d_0d_info.items()}, f, indent=4)

        # 3: the Python 1d solver (run1DBFmodel.py) is in its own repository: solver_1d_path of the user inputs

        print("1D Python files generated. Check they run properly.")
        
        return True


#XXX Finbar
class CVS1DCppGenerator(object):
    '''
    Generates Cpp files for 1D model. This is a wrapper around Lucas Muller's code.

    WARNING: THIS CPP 1D CODE IS CURRENTLY NOT OPEN SOURCE, SO THIS CLASS WON'T WORK UNTIL IT IS MADE OPEN SOURCE.
    '''


    def __init__(self, model, output_path, file_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.file_prefix = file_prefix
        self.user_resources_path = os.path.join(generators_dir_path, '../../resources')

    
    def generate_files(self):
        print("Generating 1D Cpp files...")

#XXX Finbar
class CVSCoupledCppGenerator(object):
    '''
    Generates Cpp files for coupled 0D and 1D models.
    '''
    
    def __init__(self, model, output_path, file_prefix):
        '''
        Constructor
        '''
        self.model = model
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.file_prefix = file_prefix
        self.user_resources_path = os.path.join(generators_dir_path, '../../resources')
    
    def generate_files(self):
        zeroD_generator = CVS0DCppGenerator(self.model, self.output_path, self.file_prefix)
        zeroD_generator.generate_files()

        # currently we don't generate the 1D model, we assume it is already generated.
        # oneD_generator = CVS1DCppGenerator(self.model, self.output_path, self.file_prefix)
        # oneD_generator.generate_files()
        print("Now do the coupling...")

