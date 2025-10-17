#include "tensure/utils.hpp"

/**
 * Utility: Overload the output stream operator for tsTensor.
 * @param os Output stream
 * @param tensor tsTensor to print
 * @return Output stream
 */
ostream& operator<<(ostream& os, const tsTensor& tensor) 
{
    os << tensor.str_repr ;
    return os;
}

/**
 * Find all unique indices in the given ts tensors.
 * @param taco_tensors Vecotor of tsTensor
 * @return Set of unique indices
 */
set<char> find_idxs(vector<tsTensor> taco_tensors)
{
    set<char> idxs;
    for (auto &taco_tensor : taco_tensors)
    {
        for (auto &idx : taco_tensor.idxs)
        {
            idxs.insert(idx);
        }
    }
    return idxs;
}

/**
 * Map each index to a random value between 3 and 20.
 * This is used to define the dimensions of the tensors.
 * @param idxs Set of indices
 * @return Map of index to random value
 */
map<char, int> map_id_to_val(set<char> idxs)
{
    map<char, int> id_val_map;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(3, 20);
    for (auto &idx: idxs)
    {
        id_val_map[idx] = dist(gen);
    }

    return id_val_map;
}

/**
 * Utility: Randomly return whether a tensor dimension is sparse or dense.
 * @param gen Random number generator
 * @return TensorFormat (tSparse or tDense)
 */
TensorFormat randomFormat(mt19937& gen) {
    uniform_int_distribution<int> dist(0, 1);
    return dist(gen) ? tsSparse : tsDense;
}

/**
 * Utility: Save the tsTensor metadata into a text file
 * @param tsTensor tsTensor to save the metadata
 * @param filename file name to save the metadata
 * @throw runtime_error if file cannot be opened
 * @return void
 */
void saveTensor(const tsTensor& t, const std::string& filename) {
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
    for (auto& s : t.storageFormat) out << s << " ";
    out << "\n";
}

/**
 * Utility: Load the tsTensor metadata from a text file
 * @param filename file name to load the metadata from
 * @throw runtime_error if file cannot be opened
 * @return Loaded tsTensor
 */
tsTensor loadTensor(const string& filename) {
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
    for (size_t i = 0; i < fmtCount; i++) in >> t.storageFormat[i];

    return t;
}