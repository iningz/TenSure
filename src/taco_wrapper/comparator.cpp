#include "taco_wrapper/comparator.hpp"

namespace taco_wrapper {

using namespace std;

bool compare_outputs(const string& ref_output, const string& kernel_output, double total)
{
    auto read_tensor = [](const string& path){
        unordered_map<string, double> data;
        ifstream file(path);
        if (!file.is_open()) throw runtime_error("Cannot open " + path);

        string line;
        while(getline(file, line)) {
            istringstream iss(line);
            vector<int> coords;
            string token;
            double last_val;

            // Read all but last
            while (iss >> token) {
                try {
                    last_val = stod(token);
                    break; // Last token is value
                } catch (...) {
                    coords.push_back(stoi(token));
                }
            }

            // convert coords to string key
            string key;
            for (auto &c : coords) {
                key += to_string(c) + ",";
            }
            data[key] = last_val;
        }
        return data;
    };

    auto ref_data_map = read_tensor(ref_output);
    auto out_data_map = read_tensor(kernel_output);

    if (ref_data_map.size() != out_data_map.size()) {
        return false;
    }

    for (auto& [key, val] : ref_data_map) {
        auto it = out_data_map.find(key);
        if (it == out_data_map.end()) return false;
        if (std::fabs(it->second - val) > total) return false;
    }
    
    return true;
}


}