#ifndef __H_LSTM_H__
#define __H_LSTM_H__


using namespace std;

#define LEARNING_RATE	0.0001
#define RANDOM_VALUE() ((double)rand()/RAND_MAX*2-1)	//-1~1
#define FOR(i,N) for(int i=0;i<N;++i)

typedef double DataType;

class LstmStates{

public:
    double *I_G;  // Input gate
    double *F_G;  // Forget gate
    double *O_G;  // Output gate
    double *N_I;  // New input (cell state)
    double *S;    // Cell state
    double *H;    // Hidden state (output)
    DataType *Y;  // Output value
    double *yDelta; // Error at output layer

    double *PreS;  // Previous cell state
    double *PreH;  // Previous hidden state

    LstmStates(const int hide, const int out);
    ~LstmStates();
};











class Optimizer{
private:
	double lr;
	double beta1;
	double beta2;
	double epsilon;
	double mt;
	double vt;
public:
	Optimizer(){
		//adam optimizer
		lr = 0.01;
		beta1 = 0.9;
		beta2 = 0.99;
		epsilon = 1e-8;
		mt = 0.0;
		vt = 0.0;
	};
	~Optimizer(){};
	double adam(double theta, const double dt, const int time);
	double sgd(double theta, const double dt);
};

class Delta{
	Optimizer *opt;
public:
    double data;
    Delta();
    ~Delta();
    double optimize(double theta, const int time);
};

class Deltas{
private:
    int _inNodeNum,_hideNodeNum,_outNodeNum;
public:
    Delta **dwi; // Input gate weight deltas
    Delta **dui; // Input gate recurrent deltas
    Delta *dbi;  // Input gate bias deltas
    Delta **dwf; // Forget gate weight deltas
    Delta **duf; // Forget gate recurrent deltas
    Delta *dbf;  // Forget gate bias deltas
    Delta **dwo; // Output gate weight deltas
    Delta **duo; // Output gate recurrent deltas
    Delta *dbo;  // Output gate bias deltas
    Delta **dwn; // New input weight deltas
    Delta **dun; // New input recurrent deltas
    Delta *dbn;  // New input bias deltas
    Delta **dwy; // Output layer weight deltas
    Delta *dby;  // Output layer bias deltas

    Deltas(int in, int hide, int out);
    ~Deltas();
    void resetDelta();
};

class Lstm{
public:
    Lstm(int innode, int hidenode, int outnode);  // Constructor: initializes the LSTM with input, hidden, and output node numbers
    ~Lstm();  // Destructor: cleans up memory
    void train(vector<DataType*> trainSet, vector<DataType*> labelSet, int epoche, double verification, double stopThreshold);  // Train the LSTM model
    DataType *predict(DataType *X);  // Predict function for a single input
    void showStates();  // Display the current states of the LSTM
    void showWeights();  // Display the current weights of the LSTM

private:
    LstmStates *forward(DataType *x);  // Forward pass for a single sample
    void forward(vector<DataType*> trainSet, vector<DataType*> labelSet);  // Forward pass for all samples
    void backward(vector<DataType*> trainSet, Deltas *deltaSet);  // Backpropagation for updating weights
    void optimize(Deltas *deltaSet, int epoche);  // Update weights using an optimization algorithm
    double trainLoss(vector<DataType*> x, vector<DataType*> y);  // Calculate training loss using RMSE (Root Mean Square Error)
    double verificationLoss(vector<DataType*> x, vector<DataType*> y);  // Calculate verification (or validation) loss
    void resetStates();  // Reset the states of the LSTM
    void renewWeights();  // Re-initialize weights

    int _inNodeNum;  // Number of input nodes
    int _hideNodeNum;  // Number of hidden nodes
    int _outNodeNum;  // Number of output nodes
    float _verification;  // Verification set ratio
    vector<LstmStates*> _states;  // LSTM state for each timestep
    double _learningRate;  // Learning rate for training

    double **_W_I;    // Weight matrix for input gate, connecting input to hidden layer
    double **_U_I;  // Weight matrix for input gate, connecting previous hidden state to current hidden layer
    double *_B_I;  // Bias for input gate
    double **_W_F;    // Weight matrix for forget gate, connecting input to hidden layer
    double **_U_F;  // Weight matrix for forget gate, connecting previous hidden state to current hidden layer
    double *_B_F;  // Bias for forget gate
    double **_W_O;    // Weight matrix for output gate, connecting input to hidden layer
    double **_U_O;  // Weight matrix for output gate, connecting previous hidden state to current hidden layer
    double *_B_O;  // Bias for output gate
    double **_W_G;    // Weight matrix for new memory generation, connecting input to hidden layer
    double **_U_G;  // Weight matrix for new memory generation, connecting previous hidden state to current hidden layer
    double *_B_G;  // Bias for new memory generation
    double **_W_Y;   // Weight matrix for output layer, connecting hidden layer to output
    double *_B_Y;  // Bias for output layer
};

#endif//__H_LSTM_H__

