#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>

bool isPositive1(double x);
bool isPositive2(double x);

int main(int argc, char const *argv[]) {

  clock_t start, end;
  double timeused;
  int N = 10000000;
  start = clock();
  bool what;
  double x;
  double h = 0.5;
  for (int i = 0; i < N; i++){
    x *= (-1)*h*i;
    what = isPositive1(x);
  }
  end = clock();
  timeused = (double) (end-start)/CLOCKS_PER_SEC;
  printf("Time used = %lf\n", timeused);


  for (int i = 0; i < N; i++){
    x *= (-1)*h*i;
    what = isPositive2(x);
  }
  end = clock();
  timeused = (double) (end-start)/CLOCKS_PER_SEC;

  printf("Time used = %lf\n", timeused);


  return 0;
}

bool isPositive1(double x)
{
  return (x >= 0);
}

bool isPositive2(double x)
{
  bool state;
  /*
    This functions is equivalent to
  if (x >= 0){
    state = True;
  }
  else{
    state = False;
  }
  */
  state = x >= 0 ? true : false;
  return state;
}
