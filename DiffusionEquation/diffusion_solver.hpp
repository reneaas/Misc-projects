#ifndef DIFFUSION_EQ_HPP
#define DIFFUSION_EQ_HPP

using namespace std;

class DiffusionSolver {
private:
  /* data */
  int m_N; //Gridpoints in x and y direction
  int m_M;
  double m_r;  //r = dt/h^2
  double m_dt; //Stepsize in time
  double m_h; //Stepsize in space
  double *m_v_old;
  double *m_v_new;
  double *m_x; //Defines the gridpoints
  double m_final_time;

public:
  void Initiate(int N, double r);
  void SetInitialCondition(double f(double x, double y));
  void PrintSolution();
  void Solve(double max_time);
  void MPI_Solve(double max_time);
  double ComputeNew_v(int i, int j);
  void WriteToFile(string filename);
  void MPI_ComputeNew_v(int i, int j, int M_local, double* v_old)
};

#endif
