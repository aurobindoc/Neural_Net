#include <iostream>
#include <vector>
#include <cassert>
#include "header_files/net.h"

using namespace std;

typedef vector<Neuron> layers;

Net::Net(const vector<int> topology)	{

	error = 0.0;
	recent_avg_error = 0.0;
	recent_avg_smoothing_factor = 100.0;
	int num_layers = topology.size(), num_outputs;

	/* Add layers to net */
	for (int i = 0; i < num_layers; ++i) {
		all_layers.push_back(layers());

		/* Add neurons to layers */
		num_outputs = i == topology.size()-1? 0: topology[i+1];
		for (int j = 0; j <= topology[i]; ++j) {
			all_layers.back().push_back(Neuron(num_outputs, j));
		}
		all_layers.back().back().setOutputVal(1.0);
	}
}


void Net::feed_forward(const vector<double> &input_vals)	{

	assert(input_vals.size() == all_layers[0].size()-1);

	/* Input to the Input Layer */
	for (int i = 0; i < input_vals.size(); ++i) {
		all_layers[0][i].setOutputVal(input_vals[i]);
	}

	/* Propagate to Hidden Layer */
	for (int i = 1; i < all_layers.size(); ++i) {
		layers &prev_layer = all_layers[i-1];
		layers &layer = all_layers[i];
		for (int j = 0; j < layer.size()-1; ++j) {
			layer[j].feed_forward(prev_layer);
		}
	}

}


void Net::back_propagation(const vector<double> &target_vals)	{

	layers &output_layer = all_layers.back();

	/* Calculate the RMS error */
	error = 0.0;
	for (int i = 0; i < output_layer.size()-1; ++i) {
		double delta = target_vals[i] - output_layer[i].getOutputVal();
		error += delta*delta;
	}
	error /= (output_layer.size() - 1);
	error = sqrt(error);

	recent_avg_error = (recent_avg_error * recent_avg_smoothing_factor + error) /
			(recent_avg_smoothing_factor + 1);

	/* Calculate Output layer Gradient */
	for (int i = 0; i < output_layer.size()-1; ++i) {
		output_layer[i].calculate_output_gradients(target_vals[i]);
	}

	/* Calculate Hidden layer Gradient */
	for (int i = all_layers.size()-2; i > 0 ; --i) {
		layers &hidden_layer = all_layers[i];
		layers &next_layer = all_layers[i+1];
		for (int j = 0; j < hidden_layer.size(); ++j) {
			hidden_layer[j].calculate_hidden_gradients(next_layer);
		}
	}

	/* Updates weights for all layers from output to hidden 1 */
	for (int i = all_layers.size()-1; i > 0 ; --i) {
		layers &layer = all_layers[i];
		layers &prev_layer = all_layers[i-1];

		for (int j = 0; j < layer.size()-1; ++j) {
			layer[j].update_input_weights(prev_layer);
		}
	}

}


void Net::get_results(vector<double> &result_vals)	{

	result_vals.clear();

	for (int i = 0; i < all_layers.back().size()-1; ++i) {
		result_vals.push_back(all_layers.back()[i].getOutputVal());
	}
}
