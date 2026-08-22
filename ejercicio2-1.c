#include <stdio.h>
#include <omp.h>

int es_primo(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main() {
    int max = 20000000; // Ajusten este número si sus compus son muy lentas o muy rápidas
    int primos = 0;
    double inicio, fin;

    inicio = omp_get_wtime();
    
    // PRUEBA 1: Completen con schedule(static)

    // PRUEBA 2: Completen con schedule(dynamic, 100)
    
    #pragma omp parallel for reduction(+:primos) schedule(dynamic, 100)
    for (int i = 2; i <= max; i++) {
        if (es_primo(i)) {
            primos++;
        }
    }
    
    fin = omp_get_wtime();
    printf("Primos encontrados: %d en %f segundos\n", primos, fin - inicio);
    return 0;
}