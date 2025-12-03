#include "tensure/random_gen.hpp"

map<char, int> map_id_to_val(const std::vector<char>& idxs)
{
    std::map<char, int> id_val_map;

    if (idxs.empty()) {
        // Nothing to assign, return empty map
        return id_val_map;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    // Determine upper bound for distribution
    int upper = std::min(6, static_cast<int>(idxs.size()));

    // Ensure lower bound <= upper bound
    int lower = 3;
    if (upper < lower) {
        upper = lower; // safe fallback
    }

    std::uniform_int_distribution<int> dist(lower, upper);

    for (auto& idx : idxs) {
        id_val_map[idx] = dist(gen);
    }

    return id_val_map;
}

TensorFormat random_format(mt19937& gen) {
    uniform_int_distribution<int> dist(0, 1);
    return dist(gen) ? tsSparse : tsDense;
}

static void __fillTensorRecursive(const tsTensor& tensor, tsTensorData& tensorData, vector<int>& current_coordinate, mt19937& gen, bernoulli_distribution& insert_dist, int depth)
{
    if (depth == tensor.shape.size())
    {
        // insert the data
        uniform_real_distribution<> dist(0.0, 0.5);
        double randomValue = dist(gen);

        ostringstream oss;
        oss << std::fixed << std::setprecision(2) << randomValue;
        std::string strValue = oss.str();
        // LOG_DEBUG("Inserting data: " + strValue);

        if (insert_dist(gen)) {
            tensorData.insert(current_coordinate, stod(strValue));
        }
        return;
    }
    // LOG_DEBUG("Recursive: " + to_string(tensor.shape.size()));
    // LOG_DEBUG("Recursive depth: " + to_string(depth));
    for (size_t i = 0; i < tensor.shape[depth]; i++) {
        current_coordinate[depth] = i;
        __fillTensorRecursive(tensor, tensorData, current_coordinate, gen, insert_dist, depth+1);
    }
}

static bool tns_tensor_data_save(const tsTensor& tensor, const tsTensorData& tsData, const string& filename)
{
    if (tsData.tfmt != "tns") {
        cout << "Unsupported file format save function called: " << tsData.tfmt << "\n";
    }
    ofstream out(filename);

    if (!out) {
        cerr << "Error: could not open file " << filename << endl;
        LOG_WARN("Error: could not open file " + filename);
        return false;
    }

    // save the actual tensor data
    for (size_t i = 0; i < tsData.coordinate.size(); i++)
    {
        for (size_t j = 0; j < tsData.coordinate[i].size(); j++)
        {
            out << tsData.coordinate[i][j] << " ";
        }
        out << tsData.data[i] << "\n";
    }
    out.close();
    return true;
}

static bool ttx_tensor_data_save(const tsTensor& tensor, const tsTensorData& tsData, const string& filename)
{
    if (tsData.tfmt != "ttx") {
        cout << "Unsupported file format save function called: " << tsData.tfmt << "\n";
    }

    ofstream out(filename);
    if (!out) {
        cerr << "Error: could not open file " << filename << endl;
        LOG_WARN("Error: could not open file " + filename);
        return false;
    }

    // save the header for the ttx
    string header = "%%MatrixMarket tensor coordinate real general";
    if (tsData.coordinate[0].size() == 2)
    {
        header = "%%MatrixMarket matrix coordinate real general";
    }
    out << header << "\n";

    // save the shape of the tensor
    for (int i = 0; i < tensor.shape.size(); i++)
    {
        out << tensor.shape[i] << " ";
    }
    out << tsData.coordinate.size() << "\n";

    // save the actual tensor data
    for (size_t i = 0; i < tsData.coordinate.size(); i++)
    {
        for (size_t j = 0; j < tsData.coordinate[i].size(); j++)
        {
            out << tsData.coordinate[i][j] << " ";
        }
        out << tsData.data[i] << "\n";
    }
    out.close();
    return true;
}

/**
 * This function generate random tensor data for a given tensors and return the string of filenames for each tensors.
 * 
 * */
vector<string> generate_random_tensor_data(const vector<tsTensor>& tensors, string location, string file_name_suffix, string tfmt)
{
    vector<string> datafile_names = {};
    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution insert_dist(0.4);

    ensure_directory_exists(location);

    for (size_t i = 1; i < tensors.size(); i++)
    {
        auto &tensor = tensors[i];
        // cout << tensor.name << endl;
        tsTensorData tsData;
        tsData.tfmt = tfmt;
        vector<int> current_coordinate(tensor.shape.size());
        
        // LOG_DEBUG("Inserting data to tensor: " + std::string(1, tensor.name));
        // LOG_DEBUG("Tensor shape: " + join(tensor.shape));
        // Fill in the tensor data
        __fillTensorRecursive(tensor, tsData, current_coordinate, gen, insert_dist, 0);

        // build output file path
        string filename = location + "/" + string(1,tensor.name) + (file_name_suffix == "" ? "" : "_") + file_name_suffix + "." + tfmt;

        // Write to file
        bool is_successful = false;
        if (tfmt == "ttx") {
            is_successful = ttx_tensor_data_save(tensor, tsData, filename);
        } else if (tfmt == "tns")
        {
            is_successful = tns_tensor_data_save(tensor, tsData, filename);
        }

        if (!is_successful) {
            LOG_ERROR("Failed saving the tensor data file: " + filename);
            break;
        }
        
        cout << "Saved tensor data: " << filename << endl;
        // LOG_INFO("Generated Tensor Data: "+  filename);
        datafile_names.push_back(filename);
    }

    return datafile_names;
}

// DONE
tuple<vector<tsTensor>, std::string> generate_random_einsum(int numInputs, int maxRank)
{
    // static const std::string pool = "ijklmnopqrstu";
    static const std::string pool = "ijklmn";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> rankDist(1, maxRank);
    std::uniform_int_distribution<> idxDist(0, pool.size()-1);

    // Step 1: Generate tensors with unique indices
    vector<vector<char>> tensors(numInputs);
    std::map<char, int> idxCount;

    for (int t = 0; t < numInputs; ++t)
    {
        int rank = rankDist(gen);
        set<char> used;
        while((int)used.size() < rank)
        {
            char c = pool[idxDist(gen)];
            if (used.count(c) == 0)
            {
                tensors[t].push_back(c);
                used.insert(c);
                idxCount[c]++;
            }
        }
    }

    // Step 2: Pick some indices as output indices
    vector<char> outputIdx;
    bernoulli_distribution isOutput(0.5); // ~50% of indices become output
    for (auto &p : idxCount)
    {
        if (isOutput(gen))
        {
            outputIdx.push_back(p.first);
        }
    }

    // Step 3: For all non-output indices that appear only once, duplicate in another tensor
    for (auto &p : idxCount)
    {
        char idx = p.first;
        if (find(outputIdx.begin(), outputIdx.end(), idx) != outputIdx.end()) continue;

        if (p.second == 1)
        {
            // Find the tensor that contains it
            int srcTensor = -1;
            for (int t = 0; t < numInputs; ++t)
            {
                if (std::find(tensors[t].begin(), tensors[t].end(), idx) != tensors[t].end())
                {
                    srcTensor = t;
                    break;
                }
            }
            // Pick a different tensor to duplicate it
            int targetTensor = srcTensor;
            while (targetTensor == srcTensor) targetTensor = gen() % numInputs;
            tensors[targetTensor].push_back(idx);
            idxCount[idx]++; // now appears twice
        }
    }

    // Step 4: Build einsum string
    auto makeTensor = [&](char name, const std::vector<char>& idxs)
    {
        return std::string(1, name) + "(" + join(idxs) + ")";
    };

    auto make_tsTensor = [&](char tensor_name, vector<char> tensor_idx, string string_repr) {
        tsTensor tensor;
        tensor.idxs = tensor_idx;
        tensor.name = tensor_name;
        tensor.str_repr = string_repr;

        for (size_t i = 0; i < tensor_idx.size(); i++)
        {
            TensorFormat enum_format = random_format(gen);
            switch (enum_format)
            {
            case TensorFormat::tsDense:
                tensor.storageFormat.push_back(TensorFormat::tsDense);
                break;
            case TensorFormat::tsSparse:
                tensor.storageFormat.push_back(TensorFormat::tsSparse);
                break;
            default:
                break;
            }
        }
        return tensor;
    };

    vector<tsTensor> tsTensors = {};
    string lhs = "A(" + join(outputIdx) + ")";

    tsTensors.push_back(make_tsTensor('A', outputIdx, lhs));
    string rhs;
    for (size_t i = 0; i < numInputs; ++i)
    {
        if (i > 0) rhs += " * ";

        string tensor_str = makeTensor('B'+i, tensors[i]);
        rhs += tensor_str;

        tsTensors.push_back(make_tsTensor(('B'+i), tensors[i], tensor_str));
    }

    // Step 5: Build random shapes for all tensors
    vector<char> all_idxs = find_idxs(tsTensors);
    // std::cout << join(all_idxs) << endl;
    map<char, int> id_val_map = map_id_to_val(all_idxs);
    for (auto &tensor : tsTensors)
    {
        for (size_t i = 0; i < tensor.idxs.size(); i++)
        {
            tensor.shape.push_back(id_val_map[tensor.idxs[i]]);
        }
    }

    return {tsTensors, (lhs + " = " + rhs)};
}

bool apply_sparsity_mutation(tsKernel& kernel, mt19937& gen) {
    if (kernel.tensors.empty()) return false;

    // 1. Pick a random tensor to mutate
    uniform_int_distribution<> tensor_dist(0, kernel.tensors.size() - 1);
    int t_idx = tensor_dist(gen);
    tsTensor& tensor = kernel.tensors[t_idx];

    // 2. Generate all possible formats for this tensor's shape
    vector<vector<string>> all_formats = generate_all_formats(tensor.shape.size());
    if (all_formats.empty()) return false;

    // 3. Pick a random format
    uniform_int_distribution<> fmt_dist(0, all_formats.size() - 1);
    vector<string>& selected_fmt_str = all_formats[fmt_dist(gen)];
    
    // 4. Convert string format to your enum/struct format
    vector<TensorFormat> new_format = parseTensorFormat(selected_fmt_str);

    // 5. Check if it's actually different from current
    if (is_equal(tensor.storageFormat, new_format)) {
        return false; 
    }

    // 6. Apply mutation
    tensor.storageFormat = new_format;
    return true;
}

// --- Helper: Trim whitespace ---
string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t");
    if (string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t");
    return str.substr(first, (last - first + 1));
}

// --- Helper: Extract Tensor Name from "Name(indices)" ---
string extract_name(const string& term) {
    size_t paren_pos = term.find('(');
    if (paren_pos == string::npos) return trim(term);
    return trim(term.substr(0, paren_pos));
}

bool apply_commutativity_mutation(tsKernel& kernel, mt19937& gen) {
    tsComputation& comp = kernel.computations[0];
    string einsum_str_expr = comp.expressions;

    // Separate LHS and RHS of einsum
    size_t equal_pos = einsum_str_expr.find("=");
    if (equal_pos == string::npos) return false; // invalid format

    string lhs_str = einsum_str_expr.substr(0, equal_pos);
    string rhs_str = einsum_str_expr.substr(equal_pos + 1);

    // Split RHS by '*' delimitter
    vector<string> terms;
    stringstream ss(rhs_str);
    string segment;
    while (getline(ss, segment, '*')) {
        terms.push_back(segment);
    }

    if (terms.size() < 2) {
        return false; // Not enough terms to apply commutativity
    }

    // shuffle terms to create a new grouping
    shuffle(terms.begin(), terms.end(), gen);
    stringstream new_rhs;
    for (size_t i = 0; i < terms.size(); i++) {
        new_rhs << terms[i];
        if (i < terms.size() - 1) {
            new_rhs << " * ";
        }
    }

    comp.expressions = lhs_str + "= " + new_rhs.str();

    // Synchronize the kernel.tensors vector, just in case
    map<string, tsTensor> tensor_map;
    for (const auto& t : kernel.tensors) {
        tensor_map[string(1, t.name)] = t;
    }

    vector<tsTensor> new_tensor_list;
    string out_name = extract_name(lhs_str);
    if (tensor_map.find(out_name) != tensor_map.end()) {
        new_tensor_list.push_back(tensor_map[out_name]);
    } else {
        // Fallback: If parsing failed, maybe don't mutate vector to prevent crash
        return true;
    }

    // Add Input Tensors (RHS) in the SHUFFLED order
    for (const auto& term : terms) {
        string in_name = extract_name(term);
        if (tensor_map.find(in_name) != tensor_map.end()) {
            new_tensor_list.push_back(tensor_map[in_name]);
        }
    }

    // Replace the old vector
    // Only do this if we successfully found all tensors
    if (new_tensor_list.size() == kernel.tensors.size()) {
        kernel.tensors = new_tensor_list;
    }

    return true;
}

// vector<string> sparsity_mutation(const fs::path kernel_directory, const fs::path& original_kernel_file, tsKernel &kernel, int max_mutants)
// {
//     vector<tsTensor>& tensors = kernel.tensors;
//     vector<string> mutated_file_names = {};

//     // Generate random sparse format mutation
//     random_device rd;
//     mt19937 gen(rd());
//     bernoulli_distribution insert_dist(0.2);

//     int selected_mutants = 0;
//     map<string, vector<vector<string>>> tensor_sparsity;
//     while (selected_mutants < max_mutants)
//     {
//         int total_possible_mutants = 0;
//         for (auto &tensor : tensors) {
//             vector<vector<string>> all_formats = generate_all_formats(tensor.shape.size());
//             total_possible_mutants += all_formats.size();
//             shuffle(all_formats.begin(), all_formats.end(), gen);

//             // iterate over all possible formats
//             for (auto &format : all_formats)
//             {
//                 // randomly decide whether to insert this format or not
//                 if (insert_dist(gen))
//                 {
//                     string key = string(1,tensor.name);
//                     // check whether we have already inserted a format for this
//                     if (tensor_sparsity.count(key) > 0)
//                     { // we have already inserted one sparse format value for this tensor.
                      
//                         // check this sparse format already exists
//                         bool exists = find(tensor_sparsity[key].begin(), tensor_sparsity[key].end(), format) != tensor_sparsity[key].end();
//                         if (!exists)
//                         {
//                             // insert the new mutant
//                             tensor_sparsity[key].push_back(format);

//                             selected_mutants++;
//                         }  
//                     } else { // we are inserting the first sparse format for this tensor
//                         // directly insert the format
//                         tensor_sparsity.insert({key, {format}});
//                         selected_mutants++;
//                     }
//                 }
//             }
//             // cout << tensor << endl;
//         }
//         if (total_possible_mutants < max_mutants)
//         {
//             max_mutants = total_possible_mutants;
//             bernoulli_distribution insert_dist(1);
//         }
//     }
//     // Mutation generation completed.

//     // Create the mutations into JSON kernel files.
//     int count = 0;
//     mutated_file_names.push_back(original_kernel_file.string());
//     for (auto &tensor : tensors)
//     {
//         vector<vector<string>> formats = tensor_sparsity[string(1, tensor.name)];
//         for (auto &fmt : formats)
//         {
//             vector<TensorFormat> org_format = tensor.storageFormat;
//             vector<TensorFormat> mutated_format = parseTensorFormat(fmt);
//             if (is_equal(org_format, mutated_format))
//                 continue;
            
//             tensor.storageFormat = parseTensorFormat(fmt);
//             string mutated_file_name = kernel_directory / ("kernel" + to_string(++count) + ".json");
//             // string mutated_file_name = "./data/kernel"+to_string(++count)+".json";
//             kernel.saveJson(mutated_file_name);
//             mutated_file_names.push_back(mutated_file_name);
//             tensor.storageFormat = org_format;
//         }
//     }
    
//     return mutated_file_names;
// }

// Helper to generate a unique string key for a specific kernel state
// This prevents saving duplicate files and ensures uniqueness
string get_kernel_signature(const tsKernel& kernel) {
    string sig = "";
    for(const auto& t : kernel.tensors) {
        sig += string(1, t.name) + ":";
        for(const auto& fmt : t.storageFormat) {
            // Assuming TensorFormat has a string representation or enum value
            sig += to_string(static_cast<int>(fmt)) + ","; 
        }
        sig += "|";
    }
    // Add data type info to signature if you support data mutation
    return sig;
}

string mutate_single_unique_kernel(const fs::path& directory, const string& original_kernel_filename, MutationOperator mutation_op, set<string>& generated_signatures, int mutation_id) 
{
    fs::path full_filename = directory / original_kernel_filename;
    tsKernel original_kernel;
    original_kernel.loadJson(full_filename.string());

    // Initialize RNG
    random_device rd;
    mt19937 gen(rd());

    // Ensure the original kernel is in the history so we don't mutate back to it
    generated_signatures.insert(get_kernel_signature(original_kernel));

    // Try multiple times to find a unique mutant
    // (because we might randomly pick a format we've already generated before)
    int max_retries = 100;
    for (int attempt = 0; attempt < max_retries; ++attempt) {
        // Copy original to a temp object for mutation
        tsKernel mutant_kernel = original_kernel;
        bool mutation_success = false;

        switch (mutation_op) {
            case SPARSITY:
                mutation_success = apply_sparsity_mutation(mutant_kernel, gen);
                break;
            case COMMUTATIVITY:
                mutation_success = apply_commutativity_mutation(mutant_kernel, gen);
                break;
            default:
                break;
        }

        if (!mutation_success) continue; // Mutation logic failed, retry

        // Check uniqueness
        string new_sig = get_kernel_signature(mutant_kernel);
        if (generated_signatures.find(new_sig) == generated_signatures.end()) {
            // Unique mutant found
            // save it
            generated_signatures.insert(new_sig);
            
            fs::path new_filename = directory / ("kernel" + to_string(mutation_id) + ".json");
            fs::path abs_output_path = fs::absolute(new_filename);
            cout << "Saving mutated kernel to: " << new_filename.string() << "\n";
            mutant_kernel.saveJson(new_filename.string());

            return new_filename.string();
        }
        // If we are here, we generated a mutant we have already seen - retry
    }

    return ""; // Failed to generate a unique mutant after retries

}

MutationOperator pick_random_op(std::mt19937& gen) {
    // Create a distribution from 0 to (NUM_OPS - 1)
    std::uniform_int_distribution<> dist(0, COUNT - 1);
    
    // Generate the number and cast it back to the Enum type
    int random_int = dist(gen);
    return static_cast<MutationOperator>(random_int);
}

vector<string> mutate_equivalent_kernel(const fs::path& directory, const string& original_kernel_filename, int max_mutants)
{
    // 1. Initialize the pool of sources with just the original file
    vector<string> source_pool;
    source_pool.push_back(original_kernel_filename);

    vector<string> mutated_kernel_files;
    set<string> generated_signatures;

    fs::path full_filename = directory / original_kernel_filename;
    mutated_kernel_files.push_back(full_filename.string());
    tsKernel orig_kernel;
    orig_kernel.loadJson(full_filename.string());
    generated_signatures.insert(get_kernel_signature(orig_kernel));

    random_device rd;
    mt19937 gen(rd());

    int safeguard_limit = max_mutants * 10; // to prevent infinite loops
    for (int mutation_id = 1; mutation_id <= max_mutants; ++mutation_id) {

        // Pick a random parent kernel from the pool to mutate
        uniform_int_distribution<> dist(0, source_pool.size() - 1);
        string parent_file_name = source_pool[dist(gen)];
        string full_mutated_file_name = mutate_single_unique_kernel(directory, parent_file_name, pick_random_op(gen), generated_signatures, mutation_id);
        if (!full_mutated_file_name.empty()) {
            mutated_kernel_files.push_back(full_mutated_file_name);

            // Add the new mutant to the pool
            source_pool.push_back(fs::path(full_mutated_file_name).filename().string());
        } else {
            // No more unique mutants can be generated
            mutation_id--;
            if (--safeguard_limit <= 0) {
                LOG_WARN("Reached safeguard limit while mutating kernels. Stopping further mutations.");
                break;
            }
        }
    }

    // cout << kernel << endl;
    return mutated_kernel_files;
}