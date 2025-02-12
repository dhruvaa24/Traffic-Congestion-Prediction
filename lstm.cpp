#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "lstm.h"

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
//derivate
double dsigmoid(double y){
    return y * (1.0 - y);  
}           
double dtanh(double y){
    y = tanh(y);
    return 1.0 - y * y;  
}

/*
Weight initialization
Parameters:
w - 2D weight matrix
x - number of rows
y - number of columns
*/
void initW(double **w, int x, int y){
    FOR(i, x){
        FOR(j, y)
            w[i][j] = RANDOM_VALUE();  //-1~1
    }
}


//print states of cells for debugging
void Lstm::showStates(){
	FOR(s, _states.size()){
		cout<<"states["<<s<<"]:"<<endl<<"I_G\t\tF_G\t\tO_G\t\tN_I\t\tS\t\tH"<<endl;
		FOR(i, _hideNodeNum){
			cout<<_states[s]->I_G[i]<<"\t\t";
			cout<<_states[s]->F_G[i]<<"\t\t";
			cout<<_states[s]->O_G[i]<<"\t\t";
			cout<<_states[s]->N_I[i]<<"\t\t";
			cout<<_states[s]->S[i]<<"\t\t";
			cout<<_states[s]->H[i]<<"\n";
		}
		cout<<"Y:";
		FOR(i, _outNodeNum){
			cout<<_states[s]->Y[i]<<"\t";
		}
		cout<<endl;
	}
}

// clear the cell states
void Lstm::resetStates(){
	FOR(i, _states.size()){
		delete _states[i];
	}
	_states.clear();
}

// show weights
void Lstm::showWeights(){
	cout<<"--------------------Wx+b=Y-----------------"<<endl;
	FOR(i, _outNodeNum){
    	cout<<"_W_Y:\n";
    	FOR(j, _hideNodeNum){
    		cout<<_W_Y[j][i]<<"\t";
    	}
    	cout<<"\n_BY:\n"<<_B_Y[i];
    }

    cout<<"\n\n-------------------------Wx+Uh+b=Y----------------------------"<<endl;
    FOR(j, _hideNodeNum){
    	cout<<"\n------------------\nU_:\n";
    	FOR(k, _hideNodeNum){
    		cout<<_U_I[k][j]<<"|"<<_U_F[k][j]<<"|"<<_U_O[k][j]<<"|"<<_U_G[k][j]<<endl;
    	}
    	cout<<"\nW_:\n";
    	FOR(k, _inNodeNum){
    		cout<<_W_I[k][j]<<"|"<<_W_F[k][j]<<"|"<<_W_O[k][j]<<"|"<<_W_G[k][j]<<endl;
    	}

        cout<<"\nB_:\n";
    	cout<<_B_I[j]<<"|"<<_B_F[j]<<"|"<<_B_O[j]<<"|"<<_B_G[j]<<endl;
    }
    cout<<endl<<"---------------------------------------------------"<<endl;
}

// init weights
void Lstm::renewWeights(){
	initW(_W_I, _inNodeNum, _hideNodeNum);
    initW(_U_I, _hideNodeNum, _hideNodeNum);
    initW(_W_F, _inNodeNum, _hideNodeNum);
    initW(_U_F, _hideNodeNum, _hideNodeNum);
    initW(_W_O, _inNodeNum, _hideNodeNum);
    initW(_U_O, _hideNodeNum, _hideNodeNum);
    initW(_W_G, _inNodeNum, _hideNodeNum);
    initW(_U_G, _hideNodeNum, _hideNodeNum);
    initW(_W_Y, _hideNodeNum, _outNodeNum);

    memset(_B_I, 0, sizeof(double)*_hideNodeNum);
    memset(_B_O, 0, sizeof(double)*_hideNodeNum);
    memset(_B_G, 0, sizeof(double)*_hideNodeNum);
    memset(_B_F, 0, sizeof(double)*_hideNodeNum);
    memset(_B_Y, 0, sizeof(double)*_outNodeNum);
}


/*
Constructor with the following Parameters:
innode - number of input units (number of features)
hidenode - number of hidden units
outnode - number of output units (output dimension)
*/
Lstm::Lstm(int innode, int hidenode, int outnode){
    // number of nodes
    _inNodeNum = innode;
    _hideNodeNum = hidenode;
    _outNodeNum = outnode;
	_verification = 0;
	_learningRate = LEARNING_RATE;

    // Weight initlization using dynamic allocation
    _W_I = (double**)malloc(sizeof(double*)*_inNodeNum);
    _W_F = (double**)malloc(sizeof(double*)*_inNodeNum);
    _W_O = (double**)malloc(sizeof(double*)*_inNodeNum);
    _W_G = (double**)malloc(sizeof(double*)*_inNodeNum);
    FOR(i, _inNodeNum){
        _W_I[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _W_F[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _W_O[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _W_G[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
    }

    _U_I = (double**)malloc(sizeof(double*)*_hideNodeNum);
    _U_F = (double**)malloc(sizeof(double*)*_hideNodeNum);
    _U_O = (double**)malloc(sizeof(double*)*_hideNodeNum);
    _U_G = (double**)malloc(sizeof(double*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        _U_I[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _U_F[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _U_O[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
        _U_G[i] = (double*)malloc(sizeof(double)*_hideNodeNum);
    }

    _B_I = (double*)malloc(sizeof(double)*_hideNodeNum);
    _B_F = (double*)malloc(sizeof(double)*_hideNodeNum);
    _B_O = (double*)malloc(sizeof(double)*_hideNodeNum);
    _B_G = (double*)malloc(sizeof(double)*_hideNodeNum);

    _W_Y = (double**)malloc(sizeof(double*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        _W_Y[i] = (double*)malloc(sizeof(double)*_outNodeNum);
    }
    _B_Y = (double*)malloc(sizeof(double)*_outNodeNum);

	renewWeights();

    cout<<"Lstm instance initialized."<<endl;
}

// destructor for resetting states
Lstm::~Lstm(){
	resetStates();
    FOR(i, _inNodeNum){
        if(_W_I[i]!=NULL){
            free(_W_I[i]);
            _W_I[i]=NULL;
        }
        if(_W_F[i]!=NULL){
            free(_W_F[i]);
            _W_F[i]=NULL;
        }
        if(_W_O[i]!=NULL){
            free(_W_O[i]);
            _W_O[i]=NULL;
        }
        if(_W_G[i]!=NULL){
            free(_W_G[i]);
            _W_G[i]=NULL;
        }
    }
    if(_W_I!=NULL){
        free(_W_I);
        _W_I=NULL;
    }
    if(_W_F!=NULL){
        free(_W_F);
        _W_F=NULL;
    }
    if(_W_O!=NULL){
        free(_W_O);
        _W_O=NULL;
    }
    if(_W_G!=NULL){
        free(_W_G);
        _W_G=NULL;
    }

    FOR(i, _hideNodeNum){
        if(_U_I[i]!=NULL){
            free(_U_I[i]);
            _U_I[i]=NULL;
        }
        if(_U_F[i]!=NULL){
            free(_U_F[i]);
            _U_F[i]=NULL;
        }
        if(_U_O[i]!=NULL){
            free(_U_O[i]);
            _U_O[i]=NULL;
        }
        if(_U_G[i]!=NULL){
            free(_U_G[i]);
            _U_G[i]=NULL;
        }
    }
    if(_U_I!=NULL){
        free(_U_I);
        _U_I=NULL;
    }
    if(_U_F!=NULL){
        free(_U_F);
        _U_F=NULL;
    }
    if(_U_O!=NULL){
        free(_U_O);
        _U_O=NULL;
    }
    if(_U_G!=NULL){
        free(_U_G);
        _U_G=NULL;
    }
    if(_B_I!=NULL){
        free(_B_I);
        _B_I=NULL;
    }
    if(_B_F!=NULL){
        free(_B_F);
        _B_F=NULL;
    }
    if(_B_O!=NULL){
        free(_B_O);
        _B_O=NULL;
    }
    if(_B_G!=NULL){
        free(_B_G);
        _B_G=NULL;
    }
    FOR(i, _hideNodeNum){
        if(_W_Y[i]!=NULL){
            free(_W_Y[i]);
            _W_Y[i]=NULL;
        }
    }
    if(_W_Y!=NULL){
        free(_W_Y);
        _W_Y=NULL;
    }
    if(_B_Y!=NULL){
        free(_B_Y);
        _B_Y=NULL;
    }

    cout<<"Lstm instance has been destroyed."<<endl;
}

/*
Calculate the loss for the training set
Parameters:
x, training feature set
y, training label set
*/
double Lstm::trainLoss(vector<DataType*> x, vector<DataType*> y){
	if(x.size()<=0 || y.size()<=0 || x.size()!=y.size()) return 0;
	double rmse = 0;
	double error = 0.0;
	int len = x.size();
	len -= _verification*len;   //len of training set
	FOR(i, len){
		LstmStates *state = forward(x[i]);
		DataType *pre = state->Y;
		DataType *label = y[i];
		FOR(j, _outNodeNum){
			error += (pre[j]-label[j])*(pre[j]-label[j]);
		}
		// delete state;
		// state = NULL;
        _states.push_back(state);
	}
	rmse = error/(len*_outNodeNum);
	return rmse;
}


/*
Calculate the loss for the validation set. The parameters are the same as the previous function, and the starting index for the validation set is calculated using _verification.
Parameters:
x, training feature set
y, training label set
*/
double Lstm::verificationLoss(vector<DataType*> x, vector<DataType*> y){
	if(x.size()<=0 || y.size()<=0 || x.size()!=y.size()) return 0;
	double rmse = 0;
	double error = 0.0;
	int len = x.size();
	int start = len-_verification*len;
	if(start==len) return 0;// The size of the validation set is 0 in this case
	for(int i=start;i<len;++i){
		LstmStates *state = forward(x[i]);
		DataType *pre = state->Y;
		DataType *label = y[i];
		FOR(j, _outNodeNum){
			error += (pre[j]-label[j])*(pre[j]-label[j]);
		}
        // delete state;
        // state = NULL;
        _states.push_back(state);
	}
	rmse = error/((len-start)*_outNodeNum);
	return rmse;
}


/*
Forward propagation for a single sample
Parameters:
x, feature vector of a single sample
*/
LstmStates *Lstm::forward(DataType *x){
	if(x==NULL){
		return 0;
	}

    LstmStates *lstates = new LstmStates(_hideNodeNum, _outNodeNum);
    //LstmStates *lstates = (LstmStates*)malloc(sizeof(LstmStates));
	// memset(lstates, 0, sizeof(LstmStates));

	// State from the previous time step
	if(_states.size()>0){
		memcpy(lstates->PreS, _states[_states.size()-1]->S, sizeof(double)*_hideNodeNum);
		memcpy(lstates->PreH, _states[_states.size()-1]->H, sizeof(double)*_hideNodeNum);
	}

    // Input layer broadcasted to the hidden layer
    FOR(j, _hideNodeNum){   
        double inGate = 0.0;
        double outGate = 0.0;
        double forgetGate = 0.0;
        double newIn = 0.0;
        // double s = 0.0;

        FOR(m, _inNodeNum){
            inGate += x[m] * _W_I[m][j]; 
            outGate += x[m] * _W_O[m][j];
            forgetGate += x[m] * _W_F[m][j];
            newIn += x[m] * _W_G[m][j];
        }

        FOR(m, _hideNodeNum){
            inGate += lstates->PreH[m] * _U_I[m][j];
            outGate += lstates->PreH[m] * _U_O[m][j];
            forgetGate += lstates->PreH[m] * _U_F[m][j];
            newIn += lstates->PreH[m] * _U_G[m][j];
        }

        inGate += _B_I[j];
        outGate += _B_O[j];
        forgetGate += _B_F[j];
        newIn += _B_G[j];

        lstates->I_G[j] = sigmoid(inGate);   
        lstates->O_G[j] = sigmoid(outGate);
        lstates->F_G[j] = sigmoid(forgetGate);
        lstates->N_I[j] = tanh(newIn);

        // Calculate the current time step's state
        lstates->S[j] = lstates->F_G[j]*lstates->PreS[j]+(lstates->N_I[j]*lstates->I_G[j]);
        // Output of the current time step
        // lstates->H[j] = lstates->I_G[j]*tanh(lstates->S[j]);//!!!!!!
        lstates->H[j] = lstates->O_G[j]*tanh(lstates->S[j]);//changed
    }


    //Hidden layer propagated to the output layer
    double out = 0.0;
    FOR(i, _outNodeNum){
	    FOR(j, _hideNodeNum){
	        out += lstates->H[j] * _W_Y[j][i];
	    }
	    out += _B_Y[i];
	    // lstates->Y[i] = sigmoid(out);//Output of each unit in the output layer
	    lstates->Y[i] = out;//Output of each unit in the output layer

	}

    return lstates;
}

/*
Forward propagation, batch_size calculation not yet implemented.
Parameters:
trainSet, training feature set, vector<feature vector (length must match the number of input units)>
labelSet, training label set, vector<label vector (length must match the number of output units)>
*/
void Lstm::forward(vector<DataType*> trainSet, vector<DataType*> labelSet){
	int len = trainSet.size();
	len -= _verification*len;
	FOR(i, len){
		LstmStates *state = forward(trainSet[i]);
	    // Save the standard error with respect to the output layer's partial derivative
	    double delta = 0.0;
	    FOR(j, _outNodeNum){//
	    	// delta = (labelSet[i][j]-state->Y[j])*dsigmoid(state->Y[j]);//!!!!!!!!!
	    	// delta = (labelSet[i][j]-state->Y[j]);//changed
	    	delta = 2*(state->Y[j]-labelSet[i][j]);//loss=label^2-2*label*y+y^2;   dloss/dy=2y-2label; 
	    	state->yDelta[j] = delta;
	    }
	    _states.push_back(state);
	}
}

/*
Backward propagation, calculate the partial derivatives of each weight
Parameters:
trainSet, training feature set, vector<feature vector (length must match the number of input units)>
deltas, pointer to the object storing the partial derivatives of each weight
*/
void Lstm::backward(vector<DataType*> trainSet, Deltas *deltas){
	if(_states.size()<=0){
		cout<<"need go forward first."<<endl;
	}
    //隐含层偏差，通过当前之后一个时间点的隐含层误差和当前输出层的误差计算
    double hDelta[_hideNodeNum];  
    double *oDelta = new double[_hideNodeNum];
    double *iDelta = new double[_hideNodeNum];
    double *fDelta = new double[_hideNodeNum];
    double *nDelta = new double[_hideNodeNum];
    double *sDelta = new double[_hideNodeNum];

    //当前时间之后的一个隐藏层误差
    double *oPreDelta = new double[_hideNodeNum]; 
    double *iPreDelta = new double[_hideNodeNum];
    double *fPreDelta = new double[_hideNodeNum];
    double *nPreDelta = new double[_hideNodeNum];
    double *sPreDelta = new double[_hideNodeNum];
    double *fPreGate = new double[_hideNodeNum];

    memset(oPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(iPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(fPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(nPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(sPreDelta, 0, sizeof(double)*_hideNodeNum);
    memset(fPreGate, 0, sizeof(double)*_hideNodeNum);


    int p = _states.size()-1;
    for(; p>=0; --p){//batch=1
    	// cout<<"p="<<p<<"|size:"<<_states.size()<<endl;
        // Current hidden layer
        double *inGate = _states[p]->I_G;     // Input gate
        double *outGate = _states[p]->O_G;    // Output gate
        double *forgetGate = _states[p]->F_G; // Forget gate
        double *newInGate = _states[p]->N_I;  // New memory
        double *state = _states[p]->S;        // State value
        double *h = _states[p]->H;            // Hidden layer output value

        // Previous hidden layer
        double *preH = _states[p]->PreH;   
        double *preState = _states[p]->PreS;

        FOR(k, _outNodeNum){  // Update weights for each output unit in the network
            // Update the connection weight between the hidden layer and the output layer
            FOR(j, _hideNodeNum){
                deltas->dwy[j][k].data += _states[p]->yDelta[k] * h[j];
                // _W_Y[j][k] -= _learningRate * _states[p]->yDelta[k] * h[j];
            }
            deltas->dby[k].data += _states[p]->yDelta[k];
            // _B_Y[k] -= _learningRate * _states[p]->yDelta[k];
        }

        // Calculate the partial derivative of the objective function with respect to each hidden unit in the network
        FOR(j, _hideNodeNum){
            // Various gates and unit states in the hidden layer
            oDelta[j] = 0.0;
            iDelta[j] = 0.0;
            fDelta[j] = 0.0;
            nDelta[j] = 0.0;
            sDelta[j] = 0.0;
            hDelta[j] = 0.0;

            // Partial derivative of the objective function with respect to the hidden state
            FOR(k, _outNodeNum){
                hDelta[j] += _states[p]->yDelta[k] * _W_Y[j][k];
            }
            FOR(k, _hideNodeNum){
                hDelta[j] += iPreDelta[k] * _U_I[j][k];
                hDelta[j] += fPreDelta[k] * _U_F[j][k];
                hDelta[j] += oPreDelta[k] * _U_O[j][k];
                hDelta[j] += nPreDelta[k] * _U_G[j][k];
            }

            oDelta[j] = hDelta[j] * tanh(state[j]) * dsigmoid(outGate[j]);
            sDelta[j] = hDelta[j] * outGate[j] * dtanh(state[j]) + sPreDelta[j] * fPreGate[j];
            fDelta[j] = sDelta[j] * preState[j] * dsigmoid(forgetGate[j]);
            iDelta[j] = sDelta[j] * newInGate[j] * dsigmoid(inGate[j]);
            nDelta[j] = sDelta[j] * inGate[j] * dtanh(newInGate[j]);

            // Update the weights between the previous hidden layer and the current hidden layer
            FOR(k, _hideNodeNum){
                deltas->dui[k][j].data += iDelta[j] * preH[k];
                deltas->duf[k][j].data += fDelta[j] * preH[k];
                deltas->duo[k][j].data += oDelta[j] * preH[k];
                deltas->dun[k][j].data += nDelta[j] * preH[k];
            }

            // Update the connection weights between the input layer and the hidden layer
            FOR(k, _inNodeNum){
                deltas->dwi[k][j].data += iDelta[j] * trainSet[p][k];
                deltas->dwi[k][j].data += fDelta[j] * trainSet[p][k];
                deltas->dwo[k][j].data += oDelta[j] * trainSet[p][k];
                deltas->dwn[k][j].data += nDelta[j] * trainSet[p][k];
            }

            deltas->dbi[j].data += iDelta[j];
            deltas->dbf[j].data += fDelta[j];
            deltas->dbo[j].data += oDelta[j];
            deltas->dbn[j].data += nDelta[j];
        }

        if(p == (_states.size()-1)){
            delete  []oPreDelta;
            delete  []fPreDelta;
            delete  []iPreDelta;
            delete  []nPreDelta;
            delete  []sPreDelta;
            delete  []fPreGate;
        }

        oPreDelta = oDelta;
        fPreDelta = fDelta;
        iPreDelta = iDelta;
        nPreDelta = nDelta;
        sPreDelta = sDelta;
        fPreGate = forgetGate;
	}
    delete  []oPreDelta;
    delete  []fPreDelta;
    delete  []iPreDelta;
    delete  []nPreDelta;
    delete  []sPreDelta;

	return;
}

/*
Update the weights based on the partial derivatives of each weight.
Parameters:
deltaSet - Pointer to the object storing the partial derivatives of each weight
epoche - Current iteration count
*/
void Lstm::optimize(Deltas *deltaSet, int epoche){
    FOR(i, _outNodeNum){
    	FOR(j, _hideNodeNum){
    		_W_Y[j][i] = deltaSet->dwy[j][i].optimize(_W_Y[j][i], epoche);
    	}
    	_B_Y[i] = deltaSet->dby[i].optimize(_B_Y[i], epoche);
    }

    FOR(j, _hideNodeNum){
    	FOR(k, _hideNodeNum){
    		_U_I[k][j] = deltaSet->dui[k][j].optimize(_U_I[k][j], epoche);
    		_U_F[k][j] = deltaSet->duf[k][j].optimize(_U_F[k][j], epoche);
    		_U_O[k][j] = deltaSet->duo[k][j].optimize(_U_O[k][j], epoche);
    		_U_G[k][j] = deltaSet->dun[k][j].optimize(_U_G[k][j], epoche);
    	}
    	FOR(k, _inNodeNum){
    		_W_I[k][j] = deltaSet->dwi[k][j].optimize(_W_I[k][j], epoche);
    		_W_F[k][j] = deltaSet->dwf[k][j].optimize(_W_F[k][j], epoche);
    		_W_O[k][j] = deltaSet->dwo[k][j].optimize(_W_O[k][j], epoche);
    		_W_G[k][j] = deltaSet->dwn[k][j].optimize(_W_G[k][j], epoche);
    	}

        _B_I[j] = deltaSet->dbi[j].optimize(_B_I[j], epoche);
        _B_F[j] = deltaSet->dbf[j].optimize(_B_F[j], epoche);
        _B_O[j] = deltaSet->dbo[j].optimize(_B_O[j], epoche);
        _B_G[j] = deltaSet->dbn[j].optimize(_B_G[j], epoche);
    }
}

double _LEARNING_RATE = LEARNING_RATE; //Global learning rate for the SGD optimizer
/*
Train the network
Parameters:
trainSet - Training feature set
labelSet - Training label set
epoche - Number of iterations
verification - Proportion of the validation set
stopThreshold - Early stopping threshold; stops when the change in results between two iterations is less than this threshold
*/
void Lstm::train(vector<DataType*> trainSet, vector<DataType*> labelSet, int epoche, double verification, double stopThreshold){
	if(trainSet.size()<=0 || labelSet.size()<=0 || trainSet.size()!=labelSet.size()){
		cout<<"data set error!"<<endl;
		return;
	}


    _verification = 0;
    if(verification>0 && verification<0.5){
        _verification = verification;
    }else{
        cout<<"verification rate is invalid."<<endl;
    }

	double lastTrainRmse = 0.0;
	double lastVerRmse = 0.0;
    _LEARNING_RATE = LEARNING_RATE; // Initialize the learning rate before starting training, suitable for the SGD optimizer

    // Calculate the average value for the validation set
    double verificationAvg = 0.0;
    if(_verification>0){
        int verLen = _verification*labelSet.size();
        FOR(i, verLen){            
        	verificationAvg += labelSet[labelSet.size()-verLen+i][0];
        }
        verificationAvg /= verLen;
        verificationAvg = verificationAvg<0?-verificationAvg:verificationAvg;
        cout<<"---------------avg="<<verificationAvg<<endl;
    }

    Deltas *deltaSet = new Deltas(_inNodeNum, _hideNodeNum, _outNodeNum);
    cout << "deltaset initiated. starting training." << endl;
    FOR(e, epoche) {	
        // Clear unit states for each epoch
        resetStates();
        // Forward propagation
        forward(trainSet, labelSet);
        // Backward error computation and derivative calculation
        deltaSet->resetDelta(); // Reset each weight's derivative value
        backward(trainSet, deltaSet);
        // Update weights based on derivatives
        optimize(deltaSet, e);

        // Clear unit states before validation
        resetStates();
        double trainRmse = trainLoss(trainSet, labelSet);
        double verRmse = verificationLoss(trainSet, labelSet);
        cout << "epoch:" << e << "|rmse:" << trainRmse << endl;
        if (e > 0 && abs(trainRmse - lastTrainRmse) < stopThreshold) { // If the change is small enough
            cout << "training rmse got tiny diff, stop at epoch num:" << e << endl;
            break;
        }
        if (e > 0 && verRmse != 0 && (verRmse - lastVerRmse) > (verificationAvg * 0.025)) { // Stop if validation accuracy drops significantly
            // cout << "verification rmse ascend too much:" << verRmse - lastVerRmse << ", stop in epoch:" << e << endl;
            // cout << "verification rmse ascend or got tiny diff, stop in epoch:" << e << endl;
            break;
        }

        lastTrainRmse = trainRmse;
        lastVerRmse = verRmse;
    }
    deltaSet->~Deltas();
    deltaSet = NULL;

}

/*
Predicts a single sample.
Parameters:
x: The feature set of the sample to be predicted.
*/
DataType *Lstm::predict(DataType *x) {
    // cout << "predict X>" << endl;
    // FOR(i, _inNodeNum) cout << x[i] << ",";
    // cout << endl;

    LstmStates *state = forward(x);
    DataType *ret = new DataType[_outNodeNum];
    memcpy(ret, state->Y, sizeof(DataType) * _outNodeNum); // Backup the result
    // free(state);
    _states.push_back(state); // Store the unit state at the current time step
    // cout << "Y>";
    // FOR(i, _outNodeNum) cout << ret[i] << ",";
    // cout << endl;
    return ret;
}


//adam optimizer
double Optimizer::adam(double preTheta, const double dt, const int time){
	mt = beta1*mt+(1-beta1)*dt;
	vt = beta2*vt+(1-beta2)*(dt*dt);
	double mcap = mt/(1-pow(beta1, time));
	double vcap = vt/(1-pow(beta2, time));
	double theta = preTheta - (lr*mcap)/(sqrt(vcap)+epsilon);

	// cout<<"Adam-preTheta="<<preTheta<<"|mt="<<mt<<"|vt="<<vt<<"|mcap="<<mcap<<"|vcap="<<vcap<<"|time="<<time<<"|theta="<<theta<<endl;
	return theta;
}

//sgd optimizer 
double Optimizer::sgd(double preTheta, const double dt){
	double theta = preTheta - _LEARNING_RATE*dt;
	return theta;
}

// Initialize the collection of partial derivatives
Deltas::Deltas(const int in, const int hide, const int out){
    _inNodeNum = in;
    _outNodeNum = out;
    _hideNodeNum = hide;

    dwi = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    dwf = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    dwo = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    dwn = (Delta**)malloc(sizeof(Delta*)*_inNodeNum);
    FOR(i, _inNodeNum){
        dwi[i] = new Delta[_hideNodeNum];
        dwf[i] = new Delta[_hideNodeNum];
        dwo[i] = new Delta[_hideNodeNum];
        dwn[i] = new Delta[_hideNodeNum];
    }

    dui = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    duf = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    duo = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    dun = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        dui[i] = new Delta[_hideNodeNum];
        duf[i] = new Delta[_hideNodeNum];
        duo[i] = new Delta[_hideNodeNum];
        dun[i] = new Delta[_hideNodeNum];
    }

    dbi = new Delta[_hideNodeNum];
    dbf = new Delta[_hideNodeNum];
    dbo = new Delta[_hideNodeNum];
    dbn = new Delta[_hideNodeNum];

    dwy = (Delta**)malloc(sizeof(Delta*)*_hideNodeNum);
    FOR(i, _hideNodeNum){
        dwy[i] = new Delta[_outNodeNum];
    }

    dby = new Delta[_outNodeNum];

}

Deltas::~Deltas(){
    FOR(i, _inNodeNum){
        delete [] dwi[i];
        delete [] dwf[i];
        delete [] dwo[i];
        delete [] dwn[i];
    }
    free(dwi);
    free(dwf);
    free(dwo);
    free(dwn);

    FOR(i, _hideNodeNum){
        delete [] dui[i];
        delete [] duf[i];
        delete [] duo[i];
        delete [] dun[i];
    }
    free(dui);
    free(duf);
    free(duo);
    free(dun);

    FOR(i, _hideNodeNum){
        delete [] dwy[i];
    }
    free(dwy);

    delete [] dbi;
    delete [] dbf;
    delete [] dbo;
    delete [] dbn;
    delete [] dby;
}

LstmStates::LstmStates(const int hide, const int out){
    // std::cout<<"new LstmStates initialized"<<std::endl;
    I_G = (double*)malloc(sizeof(double)*hide);
    F_G = (double*)malloc(sizeof(double)*hide);
    O_G = (double*)malloc(sizeof(double)*hide);
    N_I = (double*)malloc(sizeof(double)*hide);
    S = (double*)malloc(sizeof(double)*hide);
    H = (double*)malloc(sizeof(double)*hide);
    PreS = (double*)malloc(sizeof(double)*hide);
    PreH = (double*)malloc(sizeof(double)*hide);
    Y = (DataType*)malloc(sizeof(DataType)*out);
    yDelta = (double*)malloc(sizeof(double)*out);

    memset(I_G, 0, sizeof(double)*hide);
    memset(F_G, 0, sizeof(double)*hide);
    memset(O_G, 0, sizeof(double)*hide);
    memset(N_I, 0, sizeof(double)*hide);
    memset(S, 0, sizeof(double)*hide);
    memset(H, 0, sizeof(double)*hide);
    memset(PreS, 0, sizeof(double)*hide);
    memset(PreH, 0, sizeof(double)*hide);
    memset(Y, 0, sizeof(DataType)*out);
    memset(yDelta, 0, sizeof(double)*out);
}

LstmStates::~LstmStates(){
    // std::cout<<"deleted the LstmStates"<<std::endl;
    free(I_G);
    free(F_G);
    free(O_G);
    free(N_I);
    free(S);
    free(H);
    free(PreS);
    free(PreH);
    free(Y);
    free(yDelta);
}


Delta::Delta(){
    opt = new Optimizer();
    data = 0;
}

Delta::~Delta(){
    delete opt;
}

double Delta::optimize(double theta, const int time){
    if(opt!=NULL){
        theta = opt->adam(theta, data, time+1);//time starting from 1 || using ADAM
        // theta = opt->sgd(theta, data);//time starting from 1
    }else{
        theta -= LEARNING_RATE * data;
    }

    return theta;
}

// Reset partial derivatives and save the optimizer’s parameter states
void Deltas::resetDelta(){
    FOR(i, _inNodeNum){
        FOR(j, _hideNodeNum){
            dwi[i][j].data = 0;
            dwf[i][j].data = 0;
            dwo[i][j].data = 0;
            dwn[i][j].data = 0;
        }
    }

    FOR(i, _hideNodeNum){
        FOR(j, _hideNodeNum){
            dui[i][j].data = 0;
            duf[i][j].data = 0;
            duo[i][j].data = 0;
            dun[i][j].data = 0;
        }
    }

    FOR(i, _hideNodeNum){
        FOR(j, _outNodeNum){
            dwy[i][j].data = 0;
        }
    }

    FOR(i, _hideNodeNum){
        dbi[i].data = 0;
        dbf[i].data = 0;
        dbo[i].data = 0;
        dbn[i].data = 0;
    }

    FOR(i, _outNodeNum){
        dby[i].data = 0;
    }
}


