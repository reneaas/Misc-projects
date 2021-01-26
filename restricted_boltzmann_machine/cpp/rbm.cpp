#include "rbm.hpp"


RBM::RBM(int n_visible, int n_hidden, double eta, double mom){
    n_hidden_ = n_hidden;
    n_visible_ = n_visible;
    eta_ = eta;
    mom_ = mom;

    weights_ = arma::randn<arma::mat>(n_visible_, n_hidden_)*0.01;
    visible_bias_ = arma::randn<arma::vec>(n_visible_)*0.01;
    hidden_bias_ = arma::randn<arma::vec>(n_hidden_)*0.01;


    visible_act_ = arma::vec(n_visible_).fill(0.);
    hidden_act_ = arma::vec(n_hidden_).fill(0.);
    dW_ = arma::mat(n_visible_, n_hidden_).fill(0.);
    dvb_ = arma::vec(n_visible_).fill(0.);
    dhb_ = arma::vec(n_hidden_).fill(0.);

    std::cout << "dW rows = " << dW_.n_rows << std::endl;
    std::cout << "dW cols = " << dW_.n_cols << std::endl;
}

arma::vec RBM::sigmoid(arma::vec x){
    return 1./(1 + arma::exp(-x));
}

void RBM::visible_act(){
    arma::vec u = arma::randu(n_visible_);
    visible_prob_ = sigmoid(visible_bias_ + weights_*hidden_act_);
    visible_act_ = visible_prob_ - u;
    visible_act_.transform( [](double val){return (val >= 0);} );
}

void RBM::hidden_act(){
    arma::vec u = arma::randu(n_hidden_);
    hidden_prob_ = sigmoid(hidden_bias_ + weights_.t()*visible_act_);
    hidden_act_ = hidden_prob_ - u;
    hidden_act_.transform( [](double val){return (val >= 0);} );
}

void RBM::fit(arma::mat data, int epochs, int batch_sz, int nCDsteps){
    int n_rows = data.n_rows;
    int n_cols = data.n_cols;
    nCDsteps_ = nCDsteps;

    std::default_random_engine gen;
    std::uniform_int_distribution<int> dist(0,n_cols-1);
    int batch_sz_inv = 1./batch_sz;

    for (int epoch = 0; epoch < epochs; epoch++){
        std::cout << "epoch: " << epoch << " of " << epochs << std::endl;
        for (int b = 0; b < batch_sz; b++){
            int idx = dist(gen);
            arma::vec x = data.col(idx);

            visible_act_.swap(x);
            hidden_act();

            //Positive phase
            arma::mat CDpos = visible_act_*hidden_prob_.t(); //Outer product
            arma::vec CDpos_vb = visible_act_;
            arma::vec CDpos_hb = hidden_prob_;


            for (int j = 0; j < nCDsteps; j++){
                visible_act();
                hidden_act();
            }

            arma::mat CDneg = visible_act_*hidden_prob_.t(); //outer product u*v^T
            arma::vec CDneg_vb = visible_act_;
            arma::vec CDneg_hb = hidden_prob_;

            dW_ += eta_*(CDpos - CDneg);
            dvb_ += eta_*(CDpos_vb - CDneg_vb);
            dhb_ += eta_*(CDpos_hb - CDneg_hb);

        }
        dW_ *= batch_sz_inv;
        dvb_ *= batch_sz_inv;
        dhb_ *= batch_sz_inv;

        weights_ += dW_;
        visible_bias_ += dvb_;
        hidden_bias_ += dhb_;

        dW_.fill(0.);
        dvb_.fill(0.);
        dhb_.fill(0.);
    }
}

arma::vec RBM::predict(arma::vec x){
    visible_act_ = x;
    hidden_act();
    visible_act();
    for (int k = 0; k < nCDsteps_; k++){
        hidden_act();
        visible_act();
    }

    return visible_act_;
}
