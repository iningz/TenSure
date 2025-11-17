#include "tensure/utils.hpp"

ostream& operator<<(ostream& os, const tsTensor& tensor) 
{
    os << "Tensor: " << tensor.str_repr << "\nShape: " << "[" << join(tensor.shape, ", ") << "]\nFormats: " << "[" << join(to_string(tensor.storageFormat), ", ") << "]";
    return os;
}

ostream& operator<<(ostream& os, const tsKernel& kernel) 
{
    for (auto &tensor : kernel.tensors)
    {
        os << tensor << "\n";

        std::string name(1, tensor.name);
        auto it = kernel.dataFileNames.find(name);

        if (it != kernel.dataFileNames.end())
            os << "dataFileName: " << it->second << "\n";
        else
            os << "dataFileName: [not found]\n";

    }

    os << "Computations: ";
    for (auto &computation : kernel.computations)
    {
        os << "\n\t" << computation.expressions;
    }
    return os;
}

// bool is_exist(vector<vector<string>>)

vector<char> find_idxs(vector<tsTensor> ts_tensors)
{
    vector<char> idxs;
    for (auto &ts_tensor : ts_tensors)
    {
        for (auto &idx : ts_tensor.idxs)
        {
            bool exists = find(idxs.begin(), idxs.end(), idx) != idxs.end();
            if (!exists)
                idxs.push_back(idx);
        }
    }
    return idxs;
}

string join(const vector<string>& idxs, const string delimitter) {
    string s;
    for (size_t i = 0; i < idxs.size(); ++i) {
        s += idxs[i];
        if (i + 1 < idxs.size()) s += delimitter;
    }
    return s;
}

string join(const vector<int>& idxs, const string delimitter) {
    string s;
    for (size_t i = 0; i < idxs.size(); ++i) {
        s += to_string(idxs[i]);
        if (i + 1 < idxs.size()) s += delimitter;
    }
    return s;
}

string join(const vector<char>& idxs, const string delimitter) {
    string s;
    for (size_t i = 0; i < idxs.size(); ++i) {
        s += idxs[i];
        if (i + 1 < idxs.size()) s += delimitter;
    }
    return s;
}

string join(const set<char>& chars, const string delimitter)
{
    string s;
    for (auto it = chars.begin(); it != chars.end(); ++it)
    {
        s += *it;
        if (next(it) != chars.end())
            s += delimitter;
    }
    return s;
}


void saveTensorData(const tsTensor& t, const std::string& filename) 
{
    ofstream out(filename);
    if (!out) throw runtime_error("Failed to open file for writing");

    out << t.name << "\n";
    out << t.str_repr << "\n";

    // Save idxs
    out << t.idxs.size() << " ";
    for (char c : t.idxs) out << c << " ";
    out << "\n";

    // Save storage format
    out << t.storageFormat.size() << " ";
    for (auto& s : t.storageFormat) out << to_string(s) << " ";
    out << "\n";
}

tsTensor loadTensorData(const string& filename) 
{
    ifstream in(filename);
    if (!in) throw runtime_error("Failed to open file for reading");

    tsTensor t;
    in >> t.name;
    in.ignore(); // eat newline
    getline(in, t.str_repr);

    size_t idxCount;
    in >> idxCount;
    t.idxs.resize(idxCount);
    for (size_t i = 0; i < idxCount; i++) in >> t.idxs[i];

    size_t fmtCount;
    in >> fmtCount;
    t.storageFormat.resize(fmtCount);
    for (size_t i = 0; i < fmtCount; i++)
    {
        string s;
        in >> s;
        t.storageFormat[i] = parseTensorFormat(s);
    }

    return t;
}

void saveKernelJson(const string& filename, const vector<tsTensor>& tensors, const vector<tsComputation>& computations)
{
    nlohmann::json j;

    // Add tensors
    for (const auto& t: tensors) 
    {
        j["tensors"].push_back({
            {"name", t.name},
            {"str_repr", t.str_repr},
            {"idxs", t.idxs},
            {"shape", t.shape},
            {"storageFormat", to_string(t.storageFormat)},
            {"dataFile", (string(1,t.name) + ".tns")}
        });
    }

    // Add computations
    for (const auto& c : computations)
    {
        j["computations"].push_back({
            {"expression", c.expressions}
        });
    }

    // Write to file
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        throw std::runtime_error("Cannot open file to save JSON: " + filename);
    }
    fout << j.dump(4);
    fout.close();
}

void loadKernelJson(const string& filename, map<char, tsTensor>& tensorsMap, vector<tsComputation>& computations)
{
    ifstream fin(filename);
    if (!fin.is_open()) {
        throw runtime_error("Cannot open file to read JSON: " + filename);
    }

    nlohmann::json j;
    fin >> j;
    fin.close();

    // Parse tensors
    for (auto& t : j["tensors"])
    {
        tsTensor desc;
        string s = t["name"].get<string>();
        desc.name = s.empty() ? '\0' : s[0];
        desc.shape = t["shape"].get<vector<int>>();
        desc.storageFormat = parseTensorFormat(t["storageFormat"].get<vector<string>>());
        tensorsMap[desc.name] = desc;
    }

    // Parse computations
    computations.clear();
    for (auto& c : j["computations"])
    {
        computations.push_back({c["expression"]});
    }
}


void ensure_directory_exists(const std::string& path) 
{
    try {
        if (!fs::exists(path)) {
            fs::create_directories(path);
            std::cout << "Created directory: " << path << std::endl;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
    }
}

bool generate_ref_kernel(const vector<tsTensor>& tensors, const vector<string>& computations, const vector<string>& dataFileNames, string file_name)
{
    if (tensors.size()-1 != dataFileNames.size()) return false;
    // cout << "SDSDS 1" << endl;
    tsKernel kernel;
    for (size_t i = 0; i < tensors.size(); i++)
    {
        // cout << "SDSDS 2" << i << endl;
        auto &tensor = tensors[i];
        kernel.tensors.push_back(tensor);
        if (i != 0)
            kernel.dataFileNames.insert({string(1,tensor.name),dataFileNames[i-1]});
        else
            kernel.dataFileNames.insert({string(1, tensor.name), "-"});
    }

    for (const auto& computation : computations)
    {
        // cout << "SDSDS 3" << computation << endl;
        tsComputation comp;
        comp.expressions = computation;
        kernel.computations.push_back(comp);
    }

    // Atomic write
    string tmp_name = file_name + ".tmp";
    // cout << "SDSDS 5" << endl;
    try
    {
        kernel.saveJson(tmp_name); // write to a temporary file first
        filesystem::rename(tmp_name, file_name); // atomic replacement
    }
    catch(const std::exception& e)
    {
        cerr << "generate_kernel failed: " << e.what() << std::endl;
        LOG_ERROR((std::ostringstream{} << "generate_kernel failed: " << e.what()).str());
        // Cleanupo temp if partially written
        error_code ec;
        filesystem::remove(tmp_name, ec);
        return false;
    }
    
    return true;
}

void generate_all_formats(int rank, vector<vector<string>>& out, vector<string>& current)
{
    if ((int)current.size() == rank)
    {
        out.push_back(current);
        return;
    }

    // insert Dense
    current.push_back("Dense");
    generate_all_formats(rank, out, current);
    current.pop_back();

    // insert Sparse
    current.push_back("Sparse");
    generate_all_formats(rank, out, current);
    current.pop_back();
}

vector<vector<string>> generate_all_formats(int rank)
{
    vector<vector<string>> all;
    vector<string> current;
    generate_all_formats(rank, all, current);
    return all;
}