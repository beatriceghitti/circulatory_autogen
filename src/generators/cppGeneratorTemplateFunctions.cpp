/* Functions for index-name mapping of states and variables */
std::string generateHeaderKey(const Model0d::VariableInfo& info, const std::vector<std::string>& modules)
{
    /*
    Replicates the logic in openOutputFiles to generate the exact string used in the output file headers.
    */
    std::string mod_suffix = "_module";
    std::string mod_name = std::string(info.component);
    std::string var_name = std::string(info.name);

    // Handle "parameters" component specifically (for EXTERNAL variable type only)
    if (mod_name == "parameters") {
        std::string detected_mod = "";
        for (const auto& m : modules) {
            if (var_name.find(m) != std::string::npos) {
                detected_mod = m;
                break;
            }
        }
        
        if (!detected_mod.empty()) {
            std::string var_suffix = "_" + detected_mod;
            size_t pos = var_name.rfind(var_suffix);
            if (pos != std::string::npos && pos == var_name.size() - var_suffix.size()) {
                var_name.erase(pos);
            }
            return detected_mod + "/" + var_name;
        }
        return "parameters/" + var_name; // Fallback
    }

    // Handle standard modules (strip _module)
    size_t pos = mod_name.rfind(mod_suffix);
    if (pos != std::string::npos && pos == mod_name.size() - mod_suffix.size()) {
        mod_name.erase(pos);
    }
    
    return mod_name + "/" + var_name;
}

std::string extractNameFromHeader(std::string item) 
{
    /*
    Extract module and variable name from the file header, e.g., "1: heart/q_ra[m3];" -> "heart/q_ra"
    */

    // Remove the index prefix "1: "
    size_t colonPos = item.find(':');
    if (colonPos != std::string::npos) {
        item = item.substr(colonPos + 1);
    }
    // Remove the units "[m3]"
    size_t bracketPos = item.find('[');
    if (bracketPos != std::string::npos) {
        item = item.substr(0, bracketPos);
    }
    // Trim whitespace
    item.erase(0, item.find_first_not_of(" \t\r\n"));
    item.erase(item.find_last_not_of(" \t\r\n") + 1);
    return item;
}
/* End functions */

/* Functions for import/export of model outputs */
void Model0d::openOutputFiles(std::string outDir)
{
    int status;
	const char * aux;
	aux = outDir.c_str();
	status = mkdir(aux, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    std::string mkdirError = (status != 0 && errno != EEXIST) ? strerror(errno) : ""; // an existing folder is fine

	std::string path = outDir+"sol0D_states.txt";
	outFileStates.open(path.c_str());
    path = outDir+"sol0D_variables.txt";
	outFileVars.open(path.c_str());
    if (!outFileStates.is_open() || !outFileVars.is_open()) {
        std::cerr << "0d solver :: ERROR: openOutputFiles() could not open the output files sol0D_states.txt and"
                  << " sol0D_variables.txt in " << outDir;
        if (!mkdirError.empty()) std::cerr << " (the folder could not be created: " << mkdirError << ")";
        std::cerr << std::endl;
        exit(1);
    }

    int count = 0;
    std::vector<std::string> modules_list_tmp;
    std::string mod_suffix = "_module";
    outFileStates << "# 0: t[s]; ";
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        std::string mod_name;
        std::string state_name;
        std::string units;

        idx_states_to_output.push_back(i);
        
        mod_name = std::string(STATE_INFO[i].component);
        size_t pos = mod_name.rfind(mod_suffix);
        if (pos != std::string::npos && pos == mod_name.size() - mod_suffix.size()) {
            mod_name.erase(pos);
            if (std::find(modules_list_tmp.begin(), modules_list_tmp.end(), mod_name) == modules_list_tmp.end()) {
                modules_list_tmp.push_back(mod_name);
            }
        }
        state_name = std::string(STATE_INFO[i].name);
        units = std::string(STATE_INFO[i].units);
        std::string state_string = std::to_string(count+1) + ": " 
                                    + mod_name + "/" 
                                    + state_name + "[" 
                                    + units + "]; ";
        outFileStates << state_string;
        count++;
    }
    outFileStates << std::endl;
    outFileStates.flush();

    count = 0;
    outFileVars << "# 0: t[s]; ";
    for (size_t i = 0; i < VARIABLE_COUNT; ++i) {
    
        if (VARIABLE_INFO[i].type == EXTERNAL || VARIABLE_INFO[i].type == ALGEBRAIC){
        
            std::string mod_name;
            std::string var_name;
            std::string units;

            if (std::string(VARIABLE_INFO[i].component)=="parameters"){
                var_name = std::string(VARIABLE_INFO[i].name);
                for (size_t j = 0; j < modules_list_tmp.size(); ++j) {
                    if (var_name.find(modules_list_tmp[j]) != std::string::npos) {
                        mod_name = modules_list_tmp[j];
                        break;
                    }
                }
                std::string var_suffix = "_"+mod_name;
                size_t pos = var_name.rfind(var_suffix);
                if (pos != std::string::npos && pos == var_name.size() - var_suffix.size()) {
                    var_name.erase(pos);
                }
                units = std::string(VARIABLE_INFO[i].units);
                std::string var_string = std::to_string(count+1) + ": " 
                                        + mod_name + "/" 
                                        + var_name + "[" 
                                        + units + "]; ";
                outFileVars << var_string;
                idx_vars_to_output.push_back(i);
                count++;
            } else{
                mod_name = std::string(VARIABLE_INFO[i].component);
                size_t pos = mod_name.rfind(mod_suffix);
                if (pos != std::string::npos && pos == mod_name.size() - mod_suffix.size()) {
                    mod_name.erase(pos); 
                    var_name = std::string(VARIABLE_INFO[i].name);
                    units = std::string(VARIABLE_INFO[i].units);
                    std::string var_string = std::to_string(count+1) + ": " 
                                            + mod_name + "/" 
                                            + var_name + "[" 
                                            + units + "]; ";
                    outFileVars << var_string;
                    idx_vars_to_output.push_back(i);
                    count++;
                }
            }
        }
    }
    outFileVars << std::endl;
    outFileVars.flush();
}

void Model0d::loadFromFile(const std::string& filename, double* targetArray, 
                            const Model0d::VariableInfo infoArray[], size_t infoSize,
                            bool useFirstRow) 
{
    // Initial values requested from a file (initStatePath, e.g. the output of a previous run): stop with an error if
    // they cannot be read, rather than silently starting from the default initial values.
    auto fail = [&](const std::string& message) {
        std::cerr << "0d solver :: ERROR: initial values from " << filename << ": " << message << std::endl;
        exit(1);
    };

    std::ifstream file(filename);
    if (!file.is_open()) fail("the file cannot be opened (check initStatePath, which must end with '/')");

    std::string headerLine;
    if (!std::getline(file, headerLine)) fail("the file is empty");

    // Build the list of all modules for parameter mapping
    std::vector<std::string> modules;
    std::string mod_suffix = "_module";
    for (size_t i = 0; i < STATE_COUNT; ++i) {
        std::string m = std::string(STATE_INFO[i].component);
        size_t pos = m.rfind(mod_suffix);
        if (pos != std::string::npos) {
            m.erase(pos);
            if (std::find(modules.begin(), modules.end(), m) == modules.end()) modules.push_back(m);
        }
    }

    // Build Lookup Map
    std::map<std::string, int> lookup;
    for (size_t i = 0; i < infoSize; ++i) {
        lookup[generateHeaderKey(infoArray[i], modules)] = static_cast<int>(i);
    }

    // Parse Header Columns
    std::vector<int> colToArrayIndex;
    std::vector<std::string> unknownColumns;
    std::stringstream ss(headerLine);
    std::string item;
    while (std::getline(ss, item, ';')) {
        std::string cleaned = extractNameFromHeader(item);
        if (cleaned.empty()) continue; // after the last ';' of the header: not a column
        if (cleaned.find("t[s]") != std::string::npos || cleaned == "t" || cleaned == "0") {
            colToArrayIndex.push_back(-1); // "time" is not stored inside the statesLoc or varLoc arrays
        } else if (lookup.count(cleaned)) {
            colToArrayIndex.push_back(lookup[cleaned]); // match
        } else {
            colToArrayIndex.push_back(-2); // unknown: file header contains a variable name that doesn't exist in STATE_INFO or VARIABLE_INFO
            unknownColumns.push_back(cleaned);
        }
    }

    // Select the Data Row (only first or last row for now)
    std::string dataLine;
    if (useFirstRow) {
        while (std::getline(file, dataLine)) {
            if (!dataLine.empty()) break;
        }
    } else {
        std::string currentLine;
        while (std::getline(file, currentLine)) {
            if (!currentLine.empty()) dataLine = currentLine;
        }
    }
    if (dataLine.empty()) fail("the file has no data row");

    // Read all the values of the row: it must be complete, with numbers only (e.g. not the NaN of a failed run)
    std::vector<double> values;
    std::stringstream dataSS(dataLine);
    std::string token;
    while (dataSS >> token) {
        char* end = nullptr;
        double value = std::strtod(token.c_str(), &end);
        if (end == token.c_str() || *end != '\0' || !std::isfinite(value)) {
            fail("value " + token + " in column " + std::to_string(values.size()) + " of the "
                 + (useFirstRow ? "first" : "last") + " row is not a finite number");
        }
        values.push_back(value);
    }
    if (values.size() != colToArrayIndex.size()) {
        fail("the " + std::string(useFirstRow ? "first" : "last") + " row has " + std::to_string(values.size())
             + " values, but the header has " + std::to_string(colToArrayIndex.size()) + " columns");
    }

    std::vector<bool> loaded(infoSize, false);
    int loadedCount = 0;
    for (size_t colIdx = 0; colIdx < values.size(); ++colIdx) {
        int targetIdx = colToArrayIndex[colIdx];
        if (targetIdx >= 0) {
            targetArray[targetIdx] = values[colIdx];
            if (!loaded[targetIdx]) loadedCount++;
            loaded[targetIdx] = true;
        }
    }
    if (loadedCount == 0) fail("no column matches a state or variable of this model (file of another model?)");

    auto listNames = [](const std::vector<std::string>& names) {
        std::string list;
        for (size_t k = 0; k < names.size() && k < 5; ++k) list += (k ? ", " : "") + names[k];
        return names.size() > 5 ? list + ", ..." : list;
    };
    if (!unknownColumns.empty()) {
        std::cout << "0d solver :: WARNING: " << unknownColumns.size() << " columns of " << filename
                  << " are not in this model, and are ignored: " << listNames(unknownColumns) << std::endl;
    }
    if (infoArray == STATE_INFO && loadedCount < static_cast<int>(STATE_COUNT)) {
        std::vector<std::string> missing;
        for (size_t i = 0; i < STATE_COUNT; ++i) {
            if (!loaded[i]) missing.push_back(generateHeaderKey(STATE_INFO[i], modules));
        }
        std::cout << "0d solver :: WARNING: " << missing.size() << " states are not in " << filename
                  << ", and keep their default initial values: " << listNames(missing) << std::endl;
    }
    std::cout << "Successfully loaded " << loadedCount << " values from " << filename << std::endl;
}

void Model0d::writeOutput(double voiLoc)
{
    outFileStates << std::scientific << std::setprecision(18) << voiLoc << " ";
    for (size_t i = 0; i < idx_states_to_output.size(); ++i) {
		outFileStates << std::scientific << std::setprecision(18) << states[idx_states_to_output[i]] << " ";
    }
	outFileStates << std::endl;
    outFileStates.flush();

    outFileVars << std::scientific << std::setprecision(18) << voiLoc << " ";
    for (size_t i = 0; i < idx_vars_to_output.size(); ++i) {
		outFileVars << std::scientific << std::setprecision(18) << variables[idx_vars_to_output[i]] << " ";
    }
	outFileVars << std::endl;
    outFileVars.flush();
}

void Model0d::closeOutputFiles()
{
    outFileStates.close();
    outFileVars.close();
}
/* End functions */

/* Functions handling the MPI coupling with the 1d solver */
int Model0d::initCoupling(int couplerRank, double T0, double pre_time, double sim_time)
{
    // MPI has already been initialised and the world communicator split in main0d.cpp (setup_coupling()).
    // All messages go through the coupler: see coupler_mpi_common.h for the protocol.
    coupler_rank = couplerRank;

    DATA_LENGTH = COUPLING_DATA_LENGTH;
    zero_data_dt.assign(DATA_LENGTH, 0.0);
    parent_data_dt.assign(DATA_LENGTH, 0.0);
    zero_data.assign(N1d0d*DATA_LENGTH, 0.0);   // one row of DATA_LENGTH values per 1d-0d connection
    parent_data.assign(N1d0d*DATA_LENGTH, 0.0); // one row of DATA_LENGTH values per 1d-0d connection
    parent_data_vol.assign(DATA_LENGTH, 0.0);

    // Handshake: check that the coupler agrees on the coupling layout.
    // First check its length: a coupler built from another version of coupler_mpi_common.h would
    // otherwise stop the job with an MPI truncation error, or leave values of cfg uninitialised.
    MPI_Status status;
    MPI_Probe(coupler_rank, TAG_CONFIG, MPI_COMM_WORLD, &status);
    int n = -1;
    MPI_Get_count(&status, MPI_DOUBLE, &n);
    if (n != COUPLING_CONFIG_LENGTH) {
        std::cerr << "0d solver :: Error: the coupler sent " << n << " values in the handshake instead of "
                  << COUPLING_CONFIG_LENGTH << ": the coupler and this main0d were built from different versions"
                  << " of coupler_mpi_common.h (rebuild the coupler and regenerate the model)" << std::endl;
        return 1;
    }
    double cfg[COUPLING_CONFIG_LENGTH];
    MPI_Recv(cfg, COUPLING_CONFIG_LENGTH, MPI_DOUBLE, coupler_rank, TAG_CONFIG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // PETSc time integration scheme this main0d was built with (#define SOLVER_<scheme> above, set at generation)
#if defined(SOLVER_CN)
    const int petscScheme = PETSC_SCHEME_CN;
#elif defined(SOLVER_BDF1)
    const int petscScheme = PETSC_SCHEME_BDF1;
#elif defined(SOLVER_BDF2)
    const int petscScheme = PETSC_SCHEME_BDF2;
#else
    const int petscScheme = PETSC_SCHEME_NONE; // not built with the PETSc ODE solver
#endif
    const double mine[COUPLING_CONFIG_LENGTH] = {static_cast<double>(N1d0d), static_cast<double>(couple_volume_sum), T0, pre_time, sim_time,
                                                 tolTime, coupling_save_time(pre_time), coupling_end_time(pre_time, sim_time),
                                                 static_cast<double>(petscScheme)};
    // the PETSc scheme (last value) must be the same in the coupler configuration file, in this main0d and in the 1d solver
    const int schemeCoupler = static_cast<int>(cfg[COUPLING_CONFIG_LENGTH-1]);
    if (schemeCoupler != petscScheme) {
        std::cerr << "0d solver :: ERROR: PETSc scheme mismatch: the coupler configuration file uses "
                  << coupling_petsc_scheme_name(schemeCoupler) << ", but this main0d was built with "
                  << coupling_petsc_scheme_name(petscScheme) << ": regenerate the model, so that the coupler configuration"
                  << " file (ODEsolver, PETSc_solver), the 0d solver and the 1d solver (PETSc_solver in input.ini)"
                  << " use the same scheme (petsc_solver in solver_info of the user inputs)" << std::endl;
        return 1;
    }
    for (int j = 0; j < COUPLING_CONFIG_LENGTH-1; ++j) {
        if (std::abs(cfg[j]-mine[j]) > 1e-12*std::abs(mine[j])) {
            std::cerr << "0d solver :: Error: coupling mismatch with the coupler [N1d0d, couple_volume_sum, T0, pre_time, sim_time, tolTime, tSave, tEnd]: coupler {";
            for (int k = 0; k < COUPLING_CONFIG_LENGTH-1; ++k) std::cerr << (k ? ", " : "") << cfg[k];
            std::cerr << "} vs 0d solver {";
            for (int k = 0; k < COUPLING_CONFIG_LENGTH-1; ++k) std::cerr << (k ? ", " : "") << mine[k];
            std::cerr << "}" << std::endl;
            return 1;
        }
    }
    std::cout << "0d solver :: MPI coupling initialised, coupler is world rank " << coupler_rank << std::endl;

    return 0;
}

void Model0d::finaliseCoupling()
{
    // Tell the coupler that the 0d solver has completed its time loop.
    zero_data_dt[0] = voi;
    zero_data_dt[1] = 0.0;
    MPI_Send(zero_data_dt.data(), DATA_LENGTH, MPI_DOUBLE, coupler_rank, TAG_DONE, MPI_COMM_WORLD);

    std::cout << "### 0d solver :: MPI coupling finalised. ###" << std::endl;
}
/* End functions */