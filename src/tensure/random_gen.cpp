#include "tensure/random_gen.hpp"

map<char, int> map_id_to_val(vector<char> idxs)
{
    map<char, int> id_val_map;
    random_device rd;
    mt19937 gen(rd());
    int mini = std::min(6, static_cast<int>(idxs.size()));
    uniform_int_distribution<int> dist(3, mini);
    for (auto &idx: idxs)
    {
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

/**
 * This function generate random tensor data for a given tensors and return the string of filenames for each tensors.
 * 
 * */
vector<string> generate_random_tensor_data(const vector<tsTensor>& tensors, string location, string file_name_suffix)
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
        vector<int> current_coordinate(tensor.shape.size());
        
        // LOG_DEBUG("Inserting data to tensor: " + std::string(1, tensor.name));
        // LOG_DEBUG("Tensor shape: " + join(tensor.shape));
        // Fill in the tensor data
        __fillTensorRecursive(tensor, tsData, current_coordinate, gen, insert_dist, 0);

        // build output file path
        string filename = location + "/" + string(1,tensor.name) + (file_name_suffix == "" ? "" : "_") + file_name_suffix +".tns";

        // Write to file
        ofstream out(filename);
        if (!out) {
            cerr << "Error: could not open file " << filename << endl;
            // file_name_suffix = (file_name_suffix == "") ? "_2" : (file_name_suffix+file_name_suffix);
            continue;
        }

        // serialize
        for (size_t i = 0; i < tsData.coordinate.size(); i++)
        {
            for (size_t j = 0; j < tsData.coordinate[i].size(); j++)
            {
                out << tsData.coordinate[i][j] << " ";
            }
            out << tsData.data[i] << "\n";
        }

        out.close();
        cout << "Saved tensor data: " << filename << endl;
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

vector<string> sparsity_mutation(const fs::path kernel_directory, const fs::path& original_kernel_file, tsKernel &kernel, int max_mutants)
{
    vector<tsTensor>& tensors = kernel.tensors;
    vector<string> mutated_file_names = {};

    // Generate random sparse format mutation
    random_device rd;
    mt19937 gen(rd());
    bernoulli_distribution insert_dist(0.2);

    int selected_mutants = 0;
    map<string, vector<vector<string>>> tensor_sparsity;
    while (selected_mutants < max_mutants)
    {
        int total_possible_mutants = 0;
        for (auto &tensor : tensors) {
            vector<vector<string>> all_formats = generate_all_formats(tensor.shape.size());
            total_possible_mutants += all_formats.size();
            shuffle(all_formats.begin(), all_formats.end(), gen);

            // iterate over all possible formats
            for (auto &format : all_formats)
            {
                // randomly decide whether to insert this format or not
                if (insert_dist(gen))
                {
                    string key = string(1,tensor.name);
                    // check whether we have already inserted a format for this
                    if (tensor_sparsity.count(key) > 0)
                    { // we have already inserted one sparse format value for this tensor.
                      
                        // check this sparse format already exists
                        bool exists = find(tensor_sparsity[key].begin(), tensor_sparsity[key].end(), format) != tensor_sparsity[key].end();
                        if (!exists)
                        {
                            // insert the new mutant
                            tensor_sparsity[key].push_back(format);

                            selected_mutants++;
                        }  
                    } else { // we are inserting the first sparse format for this tensor
                        // directly insert the format
                        tensor_sparsity.insert({key, {format}});
                        selected_mutants++;
                    }
                }
            }
            // cout << tensor << endl;
        }
        if (total_possible_mutants < max_mutants)
        {
            max_mutants = total_possible_mutants;
            bernoulli_distribution insert_dist(1);
        }
    }
    // Mutation generation completed.

    // Create the mutations into JSON kernel files.
    int count = 0;
    mutated_file_names.push_back(original_kernel_file.string());
    for (auto &tensor : tensors)
    {
        vector<vector<string>> formats = tensor_sparsity[string(1, tensor.name)];
        for (auto &fmt : formats)
        {
            vector<TensorFormat> org_format = tensor.storageFormat;
            vector<TensorFormat> mutated_format = parseTensorFormat(fmt);
            if (is_equal(org_format, mutated_format))
                continue;
            
            tensor.storageFormat = parseTensorFormat(fmt);
            string mutated_file_name = kernel_directory / ("kernel" + to_string(++count) + ".json");
            // string mutated_file_name = "./data/kernel"+to_string(++count)+".json";
            kernel.saveJson(mutated_file_name);
            mutated_file_names.push_back(mutated_file_name);
            tensor.storageFormat = org_format;
        }
    }
    
    return mutated_file_names;
}

vector<string> mutate_equivalent_kernel(const fs::path& directory, const string& original_kernel_filename, MutationOperator mutation_operator, int max_mutants)
{
    fs::path full_filename = directory / original_kernel_filename;
    tsKernel kernel;
    kernel.loadJson(full_filename.string());

    vector<string> mutated_kernel_files;

    switch (mutation_operator)
    {
    case SPARSITY:
        mutated_kernel_files = sparsity_mutation(directory, full_filename, kernel, max_mutants);
        break;
    default:
        break;
    }

    // cout << kernel << endl;
    return mutated_kernel_files;
}