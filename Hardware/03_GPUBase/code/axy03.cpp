
void fun_axy(int n, double alpha, double *x, double *y)
{
    Parallel for(int i = 0; i < n; i++)
    {
        y[i] = alpha * x[i] + y[i];
    }
}

