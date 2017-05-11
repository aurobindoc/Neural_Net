#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include "header_files/trainer.h"

using namespace std;

Trainer::Trainer(const std::string& filename)
{
	train_data_file.open(filename.c_str());
}

void Trainer::get_topology(vector<int> &topology)
{
    string line;
    string label;

    getline(train_data_file, line);
    stringstream ss(line);
    ss >> label;

    if (this->isEof() || label != "topology:")
        abort();

    while (!ss.eof())
    {
        int n;
        ss >> n;
        topology.push_back(n);
    }
}

int Trainer::getNextInputs(vector<double> &input_vals)
{
	input_vals.clear();

    string str_line;
    getline(train_data_file, str_line);
    stringstream sstr(str_line);

    string str_label;
    sstr >> str_label;
    if (str_label == "in:")
    {
        double inp;

        while (sstr >> inp)
        	input_vals.push_back(inp);
    }

    return input_vals.size();
}

int Trainer::getTargetOutputs(vector<double> &target_vals)
{
	target_vals.clear();

    string str_line;
    getline(train_data_file, str_line);
    stringstream sstr(str_line);

    string str_label;
    sstr >> str_label;
    if (str_label == "out:")
    {
        double inp;

        while (sstr >> inp)
        	target_vals.push_back(inp);
    }

    return target_vals.size();
}
