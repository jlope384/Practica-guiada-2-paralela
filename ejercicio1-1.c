#include <stdio.h>
#include <omp.h>

int main() {
    double v0 = 15.5; // Velocidad inicial base
    int num_planetas = 10;
    double gravedades[10] = {9.8, 3.7, 8.87, 1.62, 24.79, 10.44, 8.87, 11.15, 274.0, 0.58}; // Distintas gravedades
    double tiempo = 5.0; 

    // PRUEBA 1: Coloquen private(v0) en la línea de abajo y vean el error en la impresión.

    // PRUEBA 2: Cambien a firstprivate(v0) y vean cómo se corrige.
    
    #pragma omp parallel for firstprivate(v0)
    for(int i = 0; i < num_planetas; i++) {
        // v0 se usa y se modifica localmente
        v0 = v0 + (gravedades[i] * tiempo); 
        printf("Hilo %d, Planeta %d: Velocidad final = %.2f\n", omp_get_thread_num(), i, v0);
    }
    return 0;
}