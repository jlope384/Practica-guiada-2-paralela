/* ------------------------------------------------------------------------
 * Universidad del Valle de Guatemala
 * Curso: CC3069 - Computacion Paralela y Distribuida
 * Ejercicio complementario: benchmark de actualizacion de particulas
 * comparando schedule(static), schedule(dynamic) y schedule(guided).
 *
 * No incluye OpenGL: solo se mide el ciclo de actualizacion con
 * omp_get_wtime(), repetido REPETICIONES veces, tal como pide el
 * enunciado. La planificacion se controla en tiempo de ejecucion con
 * schedule(runtime) + la variable de entorno OMP_SCHEDULE, para poder
 * probar static / dynamic / guided y distintos tamanos de chunk sin
 * recompilar:
 *
 *   OMP_SCHEDULE=static     ./particulas_benchmark
 *   OMP_SCHEDULE=dynamic    ./particulas_benchmark
 *   OMP_SCHEDULE=guided     ./particulas_benchmark
 *   OMP_SCHEDULE=static,100 ./particulas_benchmark
 *   OMP_SCHEDULE=dynamic,100 ./particulas_benchmark
 *   OMP_SCHEDULE=guided,100 ./particulas_benchmark
 * -------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_PARTICULAS   100000
#define REPETICIONES     1000

/* Region "costosa": las particulas que caen aqui hacen trabajo extra,
 * lo que genera desbalance de carga entre hilos (igual que en clase). */
#define REGION_XMIN  -0.3f
#define REGION_XMAX   0.3f
#define REGION_YMIN  -0.3f
#define REGION_YMAX   0.3f
#define ITERACIONES_EXTRA 40

typedef struct {
    float x, y, vx, vy;
} Particula;

Particula particulas[NUM_PARTICULAS];

void inicializarParticulas() {
    srand(42); /* semilla fija para que todas las corridas partan del mismo estado */
    for (int i = 0; i < NUM_PARTICULAS; i++) {
        particulas[i].x  = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
        particulas[i].y  = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
        particulas[i].vx = ((float) rand() / RAND_MAX) * 0.01f - 0.005f;
        particulas[i].vy = ((float) rand() / RAND_MAX) * 0.01f - 0.005f;
    }
}

static inline int enRegionCostosa(const Particula *p) {
    return (p->x >= REGION_XMIN && p->x <= REGION_XMAX &&
            p->y >= REGION_YMIN && p->y <= REGION_YMAX);
}

static inline void calculoCostoso(Particula *p) {
    float acumulado = 0.0f;
    for (int k = 1; k <= ITERACIONES_EXTRA; k++) {
        acumulado += sinf(p->x * k) * cosf(p->y * k);
    }
    p->vx += acumulado * 1e-9f;
    p->vy += acumulado * 1e-9f;
}

int main() {
    omp_set_num_threads(4);

    /* Si no se definio OMP_SCHEDULE, usar static por defecto. */
    if (getenv("OMP_SCHEDULE") == NULL) {
        omp_set_schedule(omp_sched_static, 0);
    }

    inicializarParticulas();

    double inicio = omp_get_wtime();

    for (int r = 0; r < REPETICIONES; r++) {
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < NUM_PARTICULAS; i++) {
            particulas[i].x += particulas[i].vx;
            particulas[i].y += particulas[i].vy;

            if (particulas[i].x >= 1.0f || particulas[i].x <= -1.0f) {
                particulas[i].vx = -particulas[i].vx;
            }
            if (particulas[i].y >= 1.0f || particulas[i].y <= -1.0f) {
                particulas[i].vy = -particulas[i].vy;
            }

            if (enRegionCostosa(&particulas[i])) {
                calculoCostoso(&particulas[i]);
            }
        }
    }

    double fin = omp_get_wtime();

    omp_sched_t tipoPlanificacion;
    int chunk;
    omp_get_schedule(&tipoPlanificacion, &chunk);
    const char *nombre = "desconocida";
    switch (tipoPlanificacion & ~omp_sched_monotonic) {
        case omp_sched_static:  nombre = "static";  break;
        case omp_sched_dynamic: nombre = "dynamic"; break;
        case omp_sched_guided:  nombre = "guided";  break;
        case omp_sched_auto:    nombre = "auto";    break;
        default: break;
    }

    printf("schedule=%s chunk=%d hilos=%d particulas=%d repeticiones=%d\n",
           nombre, chunk, omp_get_max_threads(), NUM_PARTICULAS, REPETICIONES);
    printf("Tiempo: %f\n", fin - inicio);

    return 0;
}
