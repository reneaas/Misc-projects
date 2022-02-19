#include "diffusion2d.hpp"


Diffusion2D::Diffusion2D(int n, double r) : DiffusionBase(n, r){

    v_old_ = arma::mat(n+1, n+1).fill(0.);
    v_new_ = arma::mat(n+1, n+1).fill(0.);
    y_ = x_;
}

/*
It's assumed that the solution is zero at the boundaries.
*/
void Diffusion2D::init_cond(double f(double x, double y)){
    for (int j = 1; j < n_; j++){
        for (int i = 1; i < n_; i++){
            v_old_.at(i, j) = f(x_(i), y_(j));
        }
    }
}


double Diffusion2D::advance(int i, int j){
    return (1-4*r_)*v_old_.at(i,j)
            + r_*(v_old_.at(i+1,j) + v_old_.at(i-1,j) + v_old_.at(i,j+1) + v_old_.at(i,j-1));
}

void Diffusion2D::solve(double tot_time){
    double t = 0;

    #ifdef _OPENMP
    {
        while (t < tot_time){
            #pragma omp parallel for
            for (int j = 1; j < n_; j++){

                for (int i = 1; i < n_; i++){
                    v_new_.at(i, j) = advance(i, j);
                }
            }
            v_old_.swap(v_new_);
            t += dt_;
        }
    }
    #else
    {
        while (t < tot_time){
            for (int j = 1; j < n_; j++){
                for (int i = 1; i < n_; i++){
                    v_new_.at(i, j) = advance(i, j);
                }
            }
            v_old_.swap(v_new_);
            t += dt_;
        }
    }
    #endif
}
