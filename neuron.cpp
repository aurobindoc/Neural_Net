#include <ctime>
#include <iostream>
#include <cstdlib>
#include "header_files/neuron.h"

using namespace std;

typedef vector<Neuron> layers;

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(int num_outputs, int ind)	{
	my_index = ind;

	for (int i = 0; i < num_outputs; ++i) {
		output_weights.push_back(connection());
		output_weights.back().weight = random_generator();
		output_weights.back().delta_weight = 0;
	}
}

double Neuron::sum_dow(const layers &next_layer) const	{
	double sum =0.0;

	for (int i = 0; i < next_layer.size()-1; ++i) {
		sum += output_weights[i].weight * next_layer[i].gradient;
	}

	return sum;
}


void Neuron::feed_forward(const layers &prev_layer)	{

	/* Get the Sum of (prev layers outputs * weights) */
	double sum = 0.0;
	for (int i = 0; i < prev_layer.size(); ++i)
		sum += prev_layer[i].getOutputVal() * prev_layer[i].output_weights[my_index].weight;

	/* Apply activation function to sum to get output value */
	output_val = activation_func(sum);
}

void Neuron::calculate_output_gradients(double target_vals)	{
	double delta = target_vals - output_val;
	gradient = delta * Neuron::activation_func_derivative(output_val);
}

void Neuron::calculate_hidden_gradients(const layers &next_layer)	{
	double dow = sum_dow(next_layer);
	gradient = dow * Neuron::activation_func_derivative(output_val);
}

void Neuron::update_input_weights(layers &prev_layer)	{
	for (int i = 0; i < prev_layer.size(); ++i) {
		Neuron &neuron = prev_layer[i];

		double old_delta_weight = neuron.output_weights[my_index].delta_weight;
		double new_delta_weight = (eta * neuron.getOutputVal() * gradient) + (alpha * old_delta_weight);

		neuron.output_weights[my_index].delta_weight = new_delta_weight;
		neuron.output_weights[my_index].weight += new_delta_weight;

	}
}
