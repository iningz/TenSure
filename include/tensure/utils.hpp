#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <sstream>
#include <fstream>

using namespace std;

typedef struct tsTensor
{
    char name;
    string str_repr;
    vector<char> idxs;
    vector<string> storageFormat;
} tsTensor;

enum TensorFormat {
    tsSparse = 0,
    tsDense = 1
};