#ifndef MONTECARLOINTEGRATOR_H
#define MONTECARLOINTEGRATOR_H

class MonteCarloIntegrator {
private:
  double m_a, m_b; //Interval [a,b].
  int m_MCsamples; //Number of MC-samples.


public:
  void Initiate(double a, double b, int N_MCsamples);
  void UniformIntegration(double f(double x));
};

#endif
