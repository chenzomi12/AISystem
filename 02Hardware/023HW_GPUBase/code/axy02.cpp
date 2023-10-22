
void fun_axy(int n, double alpha, double *x, double *y)
{
    for(int i = 0; i < n; i += 8)
    {
        y[i+0] = alpha * x[i+0] + y[i+0];
        y[i+1] = alpha * x[i+1] + y[i+1];
        y[i+2] = alpha * x[i+2] + y[i+2];
        y[i+3] = alpha * x[i+3] + y[i+3];
        y[i+4] = alpha * x[i+4] + y[i+4];
        y[i+5] = alpha * x[i+5] + y[i+5];
        y[i+6] = alpha * x[i+6] + y[i+6];
        y[i+7] = alpha * x[i+7] + y[i+7];
    }
}

