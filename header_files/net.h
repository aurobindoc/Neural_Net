#include <vector>
#include "neuron.h"

using namespace std;

typedef vector<Neuron> layers;

class Net	{
public:
	Net(const vector<int> topology);
	void feed_forward(const vector<double> &input_vals);
	void back_propagation(const vector<double> &target_vals);
	void get_results(vector<double> &result_vals);

	double getError() const { return error; }
	double get_recent_avg_error() const { return recent_avg_error; }

private:
	vector<layers> all_layers;
	double error;
	double recent_avg_error;
	double recent_avg_smoothing_factor;
};
