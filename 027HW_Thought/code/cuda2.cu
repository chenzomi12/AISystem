
void add_matrix(float* a, float* b, float* c, int N) {
    int index;
    
    for (int i = 0; i < N; ++i){
        index = i + j * N;
        c[index] = a[index] + b[index];
    }
}

int main() {
    add_matrix(a, b, c, N);
}

