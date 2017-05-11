#include <vector>
#include <cmath>

using namespace std;

class Neuron;
typedef vector<Neuron> layers;

struct connection	{
	double weight;
	double delta_weight;
};

class Neuron	{

public:
	Neuron(int num_outputs, int index);

	double sum_dow(const layers &next_layer) const;

	void feed_forward(const layers &prev_layer);
	void calculate_output_gradients(double target_vals);
	void calculate_hidden_gradients(const layers &next_layer);
	void update_input_weights(layers &prev_layer);

	double getOutputVal() const { return output_val; }
	void setOutputVal(double outputVal) { output_val = outputVal; }

private:
	int my_index;
	double output_val;
	double gradient;
	vector<connection> output_weights;

	static double eta; // Learning rate
	static double alpha; //Momentum

	static double random_generator(void)	{
		return rand()/double(RAND_MAX);
	}

	static inline double activation_func(double sum)	{
		return tanh(sum);
	}

	static inline double activation_func_derivative(double x)	{
		return 1 - x*x;
	}
};
