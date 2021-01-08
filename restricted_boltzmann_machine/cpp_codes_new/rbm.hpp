#ifndef RBM_HPP
#define RBM_HPP

#include <armadillo>

class RBM{
private:
    int n_hidden_, n_visible_, nCDsteps_, epochs_, batch_sz_, n_data_;
    double eta_, mom_;

    arma::mat weights_;
    arma::vec visible_act_, hidden_act_, visible_prob_, hidden_prob_, visible_bias_, hidden_bias_;

    arma::mat dW_;
    arma::vec dvb_, dhb_;

    arma::vec sigmoid(arma::vec x);
    void visible_act();
    void hidden_act();

public:
    RBM(int n_visible, int n_hidden);
    void fit(arma::mat data);

};

#endif
