#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <fstream>

#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

#include <nlohmann/json.hpp>

#include "tensure/formats.hpp"



using namespace std;

/**
 * Utility: Overload the output stream operator for tsTensor.
 * @param os Output stream
 * @param tensor tsTensor to print
 * @return Output stream
 */
ostream& operator<<(ostream& os, const tsTensor& tensor);

ostream& operator<<(ostream& os, const tsKernel& kernel);

/**
 * Find all unique indices in the given ts tensors.
 * @param taco_tensors Vecotor of tsTensor
 * @return Set of unique indices
 */
vector<char> find_idxs(vector<tsTensor> taco_tensors);

/**
 * Utility: Save the tsTensor metadata into a text file
 * @param tsTensor tsTensor to save the metadata
 * @param filename file name to save the metadata
 * @throw runtime_error if file cannot be opened
 * @return void
 */
void saveTensorData(const tsTensor& t, const std::string& filename);

/**
 * Utility: Load the tsTensor metadata from a text file
 * @param filename file name to load the metadata from
 * @throw runtime_error if file cannot be opened
 * @return Loaded tsTensor
 */
tsTensor loadTensorData(const string& filename);

/**
 * Utility: Save the metadata about kernel as JSON
 * @param filename file name to save the JSON file
 * @param tensors vector of tsTensors involved in the kernel
 * @param computations vector of kernel computations
 * @throw runtime_error if file cannot be opened
 * @return void
 */
void saveKernelJson(const string& filename, const vector<tsTensor>& tensors, const vector<tsComputation>& computations);

/**
 * Utility: Load the kernel and tensors from the JSON file
 * @param filename JSON file name to parse and load kernel
 * @param tensorsMap a map to load the tensor metadata for the kernel
 * @param computations a vector to load kernel computations
 * @throw runtime_error if the JSON file cannot be opened
 * @return void
 */
void loadKernelJson(const string& filename, map<char, tsTensor>& tensorsMap, vector<tsComputation>& computations);


/**
 * Utility: join indices as string (e.g., i,j,k)
 * @param idxs a vector of elements to join into a string
 * @param delimitter a delimitter to seperate the character in the string
 */
string join(const vector<char>& idxs, const string delimitter=",");
string join(const vector<int>& idxs, const string delimitter=",");
string join(const vector<string>& idxs, const string delimitter=",");
string join(const set<char>& chars, const string delimitter=",");

/**
 * Utility: ensure a directory exist, if not create one
 * @param path directory to check
 * @return void
 */
void ensure_directory_exists(const std::string& path);

/**
 * Generate kernel based on the provided tensors and computations and save the kernel file as JSON into file_name
 * @param tensors a vector tensors to use for kernel file generation.
 * @param computations a vector of computational tensor expressions in the kernel.
 * @param dataFileNames a vector of names of the data file names for each tensors (should be as same size as tensors).
 * @param file_name file name to store the kernel as JSON.
 * @return bool true of the kernel file has been successfully created, false otherwise.
 */
bool generate_ref_kernel(const vector<tsTensor>& tensors, const vector<string>& computations, const vector<string>& dataFileNames, string file_name);


vector<vector<string>> generate_all_formats(int rank);