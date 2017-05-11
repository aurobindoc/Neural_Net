#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include "header_files/net.h"
#include "header_files/trainer.h"

using namespace std;


void create_training_data_xor(string file_name, int length)	{
	ofstream myfile;
	myfile.open(file_name);

	/* Add the topology */
	myfile<<"topology: 2 3 1"<<endl;

	/* Add training data to the Neural Net */
	for (int i = 0; i < length/4; ++i) {
		myfile<<"in: 0.0 0.0"<<endl;
		myfile<<"out: 0.0"<<endl;
		myfile<<"in: 0.0 1.0"<<endl;
		myfile<<"out: 1.0"<<endl;
		myfile<<"in: 1.0 0.0"<<endl;
		myfile<<"out: 1.0"<<endl;
		myfile<<"in: 1.0 1.0"<<endl;
		myfile<<"out: 0.0"<<endl;
	}

	myfile.close();
}

void create_training_data_add(string file_name, int length)	{
	ofstream myfile;
	myfile.open(file_name);

	/* Add the topology */
	myfile<<"topology: 3 3 3"<<endl;

	/* Add training data to the Neural Net */
	for (int i = 0; i < length/4; ++i) {
		myfile<<"in: 0.0 0.0 0.0"<<endl;
		myfile<<"out: 0.0 0.0 1.0"<<endl;
		myfile<<"in: 0.0 0.0 1.0"<<endl;
		myfile<<"out: 0.0 1.0 0.0"<<endl;
		myfile<<"in: 0.0 1.0 0.0"<<endl;
		myfile<<"out: 0.0 1.0 1.0"<<endl;
		myfile<<"in: 0.0 1.0 1.0"<<endl;
		myfile<<"out: 1.0 0.0 0.0"<<endl;
		myfile<<"in: 1.0 0.0 0.0"<<endl;
		myfile<<"out: 1.0 0.0 1.0"<<endl;
		myfile<<"in: 1.0 0.0 1.0"<<endl;
		myfile<<"out: 1.0 1.0 0.0"<<endl;
		myfile<<"in: 1.0 1.0 0.0"<<endl;
		myfile<<"out: 1.0 1.0 1.0"<<endl;
		myfile<<"in: 1.0 1.0 1.0"<<endl;
		myfile<<"out: 0.0 0.0 0.0"<<endl;
	}

	myfile.close();
}



void showVectorVals(const string& prefix, const vector<double> &values)
{
    cout << prefix << " ";
    for (int i = 0; i < values.size(); ++i)
        cout << fixed << values[i] << " ";

    cout << endl;
}


int main()
{
	 create_training_data_xor("train_data_xor.txt", 2000);
//	create_training_data_add("train_data_add.txt", 2000);
	cout<<endl<<"#================ STARTING TRAINING ================#"<<endl;

    Trainer trainData("train_data.txt");

    vector<int> topology;
    trainData.get_topology(topology);

    Net myNet(topology);

    vector<double> input_vals, target_vals, result_vals;
    int trainingPass = 0;

    while (!trainData.isEof())
    {
        ++trainingPass;
        // Get new input data and feed it forward:
		if (trainData.getNextInputs(input_vals) != topology[0])
			break;

        std::cout << std::endl << "Pass " << trainingPass << std::endl;

        showVectorVals("Inputs:", input_vals);
        myNet.feed_forward(input_vals);

        // Collect the net's actual output results:
        myNet.get_results(result_vals);
        showVectorVals("Outputs:", result_vals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(target_vals);
        showVectorVals("Targets:", target_vals);

        myNet.back_propagation(target_vals);

        // Report how well the training is working, average over recent samples:
        cout << "Net current error: " << myNet.getError() << std::endl;
        cout << "Net recent average error: " << myNet.get_recent_avg_error() << endl;
    }

    cout<<endl<<"#================ TRAINING COMPLETE ================#" <<endl<<endl;


    cout<<endl<<"#================ STARTING TEST ================#"<<endl;

	int test_arr[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };

	for (int i = 0; i < 4; ++i)
	{
		input_vals.clear();
		input_vals.push_back(test_arr[i][0]);
		input_vals.push_back(test_arr[i][1]);

		myNet.feed_forward(input_vals);
		myNet.get_results(result_vals);

		showVectorVals("Inputs:", input_vals);
		showVectorVals("Outputs:", result_vals);

		cout << endl;
	}

	cout<<endl<<"#================ COMPLETE TEST ================#"<<endl;

}
