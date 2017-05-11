#include <string>
#include <vector>

using namespace std;

class Trainer	{
private:
    std::ifstream   train_data_file;

public:
    Trainer(const std::string& filename);

public: // getter/setter
    inline bool isEof(void) const { return train_data_file.eof(); }

public: // public method(s)
    void get_topology(std::vector<int> &arr_topology);

    // Returns the number of input values read from the file:
    int getNextInputs(vector<double> &arr_inputVals);
    int getTargetOutputs(vector<double> &arr_targetOutputVals);
};
