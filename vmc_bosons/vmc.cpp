#include "vmc.hpp"


VMC::VMC(int n_particles, int dims, double step_sz, double alpha, std::string sampling){
    n_particles_ = n_particles;
    dims_ = dims;
    step_sz_ = step_sz;
    alpha_ = alpha;
    omega_ = 1.;
    sampling_ = sampling;


    loc_energy = &VMC::loc_energy_no_int;
    trial_fn = &VMC::trial_fn_no_int;
    quantum_force = &VMC::quantum_force_spherical;

    if (sampling == "brute_force"){
        gen_trial_pos = &VMC::gen_trial_pos_bf;
        metropolis = &VMC::metropolis_bf;
    }
    else if (sampling == "importance_sampling"){
        gen_trial_pos = &VMC::gen_trial_pos_is;
        metropolis = &VMC::metropolis_is;
        D_ = 0.5;
        sqrt_step_sz_ = sqrt(step_sz_);
    }
}

VMC::VMC(int n_particles, int dims, double step_sz, double alpha, double beta, std::string sampling){
    n_particles_ = n_particles;
    dims_ = dims;
    step_sz_ = step_sz;
    alpha_ = alpha;
    omega_ = 1.;
    sampling_ = sampling;
    beta_ = beta;
    a_ = 0.0043;


    loc_energy = &VMC::loc_energy_with_int;
    trial_fn = &VMC::trial_fn_with_int;
    quantum_force = &VMC::quantum_force_elliptical;
    laplacian = &VMC::laplacian_with_int;

    if (sampling == "brute_force"){
        gen_trial_pos = &VMC::gen_trial_pos_bf;
        metropolis = &VMC::metropolis_bf;
    }
    else if (sampling == "importance_sampling"){
        gen_trial_pos = &VMC::gen_trial_pos_is;
        metropolis = &VMC::metropolis_is;
        D_ = 0.5;
        sqrt_step_sz_ = sqrt(step_sz_);
    }
}


void VMC::gen_trial_pos_bf(Particle *particle){
    arma::mat dr = 2*arma::randu(dims_, n_particles_) - 1;
    particle->trial_pos_ = particle->pos_ + step_sz_*dr;
}

void VMC::gen_trial_pos_is(Particle *particle){
    particle->old_force_.swap(particle->new_force_); //Swap the force matrices.
    arma::mat xi = arma::randn(dims_, n_particles_);
    particle->trial_pos_ = particle->pos_ + D_ * particle->old_force_ * step_sz_ + xi * sqrt_step_sz_;
}

double VMC::loc_energy_no_int(Particle *particle){
    double r_sq = arma::dot(particle->pos_, particle->pos_);
    return alpha_ * dims_ * n_particles_ + (0.5 * omega_ * omega_ - 2 * alpha_ * alpha_) * r_sq;
}

double VMC::trial_fn_no_int(Particle *particle){
    double r_sq = arma::dot(particle->trial_pos_, particle->trial_pos_);
    return exp(-2 * alpha_ * r_sq);
}


double VMC::greens_fn(arma::mat x, arma::mat y, arma::mat force_y){
    arma::mat diff = x - y - D_*force_y*step_sz_;
    double r = arma::dot(diff, diff);
    return exp(-r / (4 * D_ * step_sz_));
}

void VMC::quantum_force_spherical(arma::mat *pos, arma::mat *force){
    (*force) = -4 * alpha_ * (*pos);
}


void VMC::quantum_force_elliptical(arma::mat *pos, arma::mat *force){
    (*force) = -4*alpha_*(*pos);
    (*force).row(2) *= beta_;

    for (int i = 0; i < n_particles_; i++){
        arma::vec ri = (*pos).col(i);
        arma::vec tmp = (*force).col(i);
        for (int j = 0; j < n_particles_; j++){
            if (i != j){
                arma::vec rj = (*pos).col(j);
                arma::vec diff = ri - rj;
                double r_ij = arma::norm(diff);
                double dudr = a_ / (r_ij * (r_ij - a_));
                tmp += 2 * diff * dudr / r_ij;
            }
        }
        (*force).col(i) += tmp;
    }
}

/*
Metropolis sampling with importance sampling.
*/
void VMC::metropolis_is(Particle *particle, double *last_trial, double *energy){
    (this->*gen_trial_pos)(particle);
    (this->*quantum_force)(&(particle->trial_pos_), &(particle->new_force_));
    double trial = (this->*trial_fn)(particle);
    double greens_fn_old = greens_fn(particle->pos_, particle->trial_pos_, particle->new_force_);
    double greens_fn_new = greens_fn(particle->trial_pos_, particle->pos_, particle->old_force_);
    double ratio = (greens_fn_old*trial)/(greens_fn_new*(*last_trial));
    if (ratio >= 1){
        particle->pos_.swap(particle->trial_pos_);
        *energy = (this->*loc_energy)(particle);
        *last_trial = trial;
    }
    else if (arma::randu() <= ratio){
        particle->pos_.swap(particle->trial_pos_);
        *energy = (this->*loc_energy)(particle);
        *last_trial = trial;
    }
}

/*
Brute force Metropolis sampling.
*/
void VMC::metropolis_bf(Particle *particle, double *last_trial, double *energy){
    (this->*gen_trial_pos)(particle);
    double trial = (this->*trial_fn)(particle);
    double ratio = trial/(*last_trial);
    if (ratio >= 1){
        particle->pos_.swap(particle->trial_pos_);
        *energy = (this->*loc_energy)(particle);
        *last_trial = trial;
    }
    else if (arma::randu() <= ratio){
        particle->pos_.swap(particle->trial_pos_);
        *energy = (this->*loc_energy)(particle);
        *last_trial = trial;
    }
}

/*
The standard Monte Carlo simulation
*/
double VMC::monte_carlo_sim(int mc_samples, int therm_samples){
    //Set up system
    double energy, energy_sq;
    double E_mean = 0., EE_mean = 0.;
    energies_ = arma::vec(mc_samples);

    #ifdef _OPENMP
    {
        #pragma omp parallel private(energy, energy_sq) reduction(+:E_mean, EE_mean)
        {
            arma::arma_rng::set_seed(omp_get_thread_num() + 42);
            Particle particle(n_particles_, dims_, sampling_);
            energy = (this->*loc_energy)(&particle); //Initial energy of the system
            double last_trial = (this->*trial_fn)(&particle);

            //Burn-in period
            for (int n = 0; n < therm_samples; n++){
                (this->*metropolis)(&particle, &last_trial, &energy);
            }


            #pragma omp for
            for (int n = 0; n < mc_samples; n++){
                (this->*metropolis)(&particle, &last_trial, &energy);
                energy_sq = energy*energy;
                E_mean += energy;
                EE_mean += energy_sq;
                energies_.at(n) = energy;
            }
        }
        E_mean /= mc_samples;
        EE_mean /= mc_samples;
    }
    #else
    {
        Particle particle(n_particles_, dims_, sampling_);
        double last_trial = (this->*trial_fn)(&particle);
        energy = (this->*loc_energy)(&particle); //Initial energy of the system

        for (int n = 0; n < therm_samples; n++){
            (this->*metropolis)(&particle, &last_trial, &energy);
        }

        for (int n = 0; n < mc_samples; n++){
            (this->*metropolis)(&particle, &last_trial, &energy);
            energy_sq = energy*energy;
            E_mean += energy;
            EE_mean += energy_sq;
        }
        E_mean /= mc_samples;
        EE_mean /= mc_samples;
    }
    #endif

    E_mean_ = E_mean;
    EE_mean_ = EE_mean;

    return E_mean_;
}


/*
Bootstrapping of the energy
*/
void VMC::bootstrap(double *mean_energy, double *stddev, int bootstrap_samples){
    int n = energies_.n_elem;
    *mean_energy = 0.;
    arma::vec tmp, bootstrap_mean = arma::vec(bootstrap_samples);
    #ifdef _OPENMP
    {
        #pragma omp parallel private(tmp)
        {
            #pragma omp for
            for (int i = 0; i < bootstrap_samples; i++){
                arma::uvec idx = arma::randi<arma::uvec>(n, arma::distr_param(0, n-1));
                tmp = energies_(idx);
                bootstrap_mean.at(i) = arma::mean(tmp);
            }
        }
    }
    #else
    {
        for (int i = 0; i < bootstrap_samples; i++){
            arma::uvec idx = arma::randi<arma::uvec>(n, arma::distr_param(0, n-1));
            tmp = energies_(idx);
            bootstrap_mean.at(i) = arma::mean(tmp);
        }
    }
    #endif

    *mean_energy = arma::mean(bootstrap_mean);
    *stddev = arma::stddev(bootstrap_mean);
}


/*
Trial function with Jastrow factor.
*/
double VMC::trial_fn_with_int(Particle *particle){
    double jastrow = 1.;
    for (int i = 0; i < n_particles_-1; i++){
        arma::vec ri = particle->trial_pos_.col(i);
        for (int j = i+1; j < n_particles_; j++){
            arma::vec rj = particle->trial_pos_.col(j);
            arma::vec diff = ri-rj;
            double r_ij = arma::norm(diff);
            if (r_ij <= a_){
                return 0.;
            }
            jastrow *= 1 - a_ / r_ij;
        }
    }
    jastrow *= jastrow;
    arma::mat r_mat = particle->trial_pos_;
    r_mat.row(2) *= sqrt(beta_);
    double r = arma::dot(r_mat, r_mat);
    return exp(-2 * alpha_ * r) * jastrow;
}


double VMC::laplacian_with_int(Particle *particle){
    double laplacian = 0.;
    laplacian += 2 * alpha_ * (2 + beta_) * n_particles_;
    for (int i = 0; i < n_particles_; i++){
        arma::vec ri = particle->pos_.col(i);
        laplacian -= 4 * alpha_ * alpha_ 
                * (
                    ri.at(0) * ri.at(0) 
                    + ri.at(1) * ri.at(1) 
                    + beta_ * beta_ * ri.at(2) * ri.at(2)            
        );

        arma::vec grad_phi = -2 * alpha_ * ri;
        grad_phi.at(2) *= beta_;
        arma::vec grad_u = arma::vec(3).fill(0.);
        for (int j = 0; j < n_particles_; j++){
            if (i != j){
                arma::vec rj = particle->pos_.col(j);
                arma::vec diff = ri-rj;
                double r_ij = arma::norm(diff);
                double dudr = a_/(r_ij*(r_ij - a_));
                double d2udr2 = (a_ - 2*r_ij)*dudr/(r_ij*(r_ij - a_));

                grad_u += diff*dudr/r_ij;

                laplacian -= (d2udr2 + 2*dudr/r_ij);
            }

        }
        laplacian -= 2*arma::dot(grad_phi, grad_u);
        laplacian -= arma::dot(grad_u, grad_u);
    }
    return laplacian;
}


double VMC::loc_energy_with_int(Particle *particle){
    double energy = 0.;
    energy += (this->*laplacian)(particle); //Add the contribution from the laplacian part of the local energy

    //Add the contribution from the potential part of the local energy
    arma::mat r_mat = particle->pos_;
    r_mat.row(2) *= beta_;
    energy += arma::dot(r_mat, r_mat);

    return 0.5*energy;
}

/*
metropolis sampling specialized for the computation of the one body density
*/
void VMC::metropolis_one_body_density(Particle *particle, double *last_trial, double *r){
    (this->*gen_trial_pos)(particle);
    (this->*quantum_force)(&(particle->trial_pos_), &(particle->new_force_));
    double trial = (this->*trial_fn)(particle);
    double greens_fn_old = greens_fn(particle->pos_, particle->trial_pos_, particle->new_force_);
    double greens_fn_new = greens_fn(particle->trial_pos_, particle->pos_, particle->old_force_);
    double ratio = (greens_fn_old*trial)/(greens_fn_new*(*last_trial));
    if (ratio >= 1){
        particle->pos_.swap(particle->trial_pos_);
        *r = arma::norm(particle->pos_.col(0));
        *last_trial = trial;
    }
    else if (arma::randu() <= ratio){
        particle->pos_.swap(particle->trial_pos_);
        *r = arma::norm(particle->pos_.col(0));
        *last_trial = trial;
    }
}



/*
Stores L^2-norm of particle 0 in an array.
*/
void VMC::one_body_density(int mc_samples, int therm_samples, std::string filename){
    double r = 0.;
    radii_ = arma::vec(mc_samples);
    metropolis = &VMC::metropolis_one_body_density;

    #ifdef _OPENMP
    {
        #pragma omp parallel private(r)
        {
            arma::arma_rng::set_seed(omp_get_thread_num()+40);
            double mean_energy = monte_carlo_sim(mc_samples, therm_samples);
            Particle particle(n_particles_, dims_, sampling_);
            // energy = (this->*loc_energy)(&particle); //Initial energy of the system
            double last_trial = (this->*trial_fn)(&particle);

            for (int n = 0; n < therm_samples; n++){
                (this->*metropolis)(&particle, &last_trial, &r);
            }


            #pragma omp for
            for (int n = 0; n < mc_samples; n++){
                (this->*metropolis)(&particle, &last_trial, &r);
                radii_.at(n) = r;
            }
        }

    }
    #else
    {
        Particle particle(n_particles_, dims_, sampling_);
        double last_trial = (this->*trial_fn)(&particle);
        for (int n = 0; n < therm_samples; n++){
            (this->*metropolis)(&particle, &last_trial, &r);
        }

        for (int n = 0; n < mc_samples; n++){
            (this->*metropolis)(&particle, &last_trial, &r);
            radii_.at(i) = r;
        }
    }
    #endif

    //Write the result to file.
    std::ofstream ofile;
    ofile.open(filename);
    for (int i = 0; i < radii_.n_elem; i++){
        ofile << radii_.at(i) << std::endl;
    }
    ofile.close();
}



/*
Methods that write computed quantities to file.
*/
void VMC::write_to_file(std::string filename){
    std::ofstream ofile;
    ofile.open(filename);
    ofile << alpha_ << " " << E_mean_ << " " << EE_mean_ << std::endl;
    ofile.close();
}

void VMC::write_to_file_all(std::string filename){
    std::ofstream ofile;
    ofile.open(filename);
    ofile << alpha_ << std::endl;
    for (int i = 0; i < energies_.n_elem; i++){
        ofile << energies_(i) << std::endl;
    }
    ofile.close();
}
