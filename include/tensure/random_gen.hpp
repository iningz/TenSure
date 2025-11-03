#include <set>
#include <map>
#include <random>
#include "tensure/formats.hpp"
#include "tensure/utils.hpp"

#include <iostream>

using namespace std;

/**
 * Map each index to a random value between 3 and 20.
 * This is used to define the dimensions of the tensors.
 * @param idxs Set of indices
 * @return Map of index to random value
 */
map<char, int> map_id_to_val(vector<char> idxs);

/**
 * Utility: Randomly return whether a tensor dimension is sparse or dense.
 * @param gen Random number generator
 * @return TensorFormat (tSparse or tDense)
 */
TensorFormat random_format(mt19937& gen);


tuple<vector<tsTensor>, std::string> generate_random_einsum(int numInputs, int maxRank);

vector<string> generate_random_tensor_data(const vector<tsTensor>& tensors, string location, string file_name_suffix);

vector<string> mutate_equivalent_kernel(const fs::path& directory, const string& original_kernel_filename, MutationOperator mutation_operator, int max_mutants = -1);