/* ------------------------------------------------------------------------
 * Universidad del Valle de Guatemala
 * Curso: CC3069 – Computación Paralela y Distribuida
 * Sección: 30
 * Fecha: 08/12/2026
 * Descripción: ejemplo de integración de OpenMP y OpenGL: simulación
 * paralela del movimiento de partículas, con una REGIÓN COSTOSA que
 * provoca desbalance de carga entre hilos para comparar schedule(static),
 * schedule(dynamic) y schedule(guided).
 * -------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/freeglut.h>
#include <omp.h>

/* ---------------------------------------------------------
 * Parámetros generales de la simulación
 * --------------------------------------------------------- */
#define NUM_PARTICULAS   300000   /* Aumentado para tener suficientes iteraciones a repartir */
#define DURACION_PRUEBA  10.0

/* ---------------------------------------------------------
 * Región costosa (rectángulo dentro de la pantalla, en
 * coordenadas normalizadas [-1, 1], igual que la posición
 * de las partículas). Ver diagrama en el enunciado.
 * --------------------------------------------------------- */
#define REGION_XMIN  -0.3f
#define REGION_XMAX   0.3f
#define REGION_YMIN  -0.3f
#define REGION_YMAX   0.3f

/* Cantidad de "trabajo" adicional que se realiza por cada
 * partícula que cae dentro de la región costosa. Entre más
 * alto, mayor el desbalance de carga. Ajustar según se
 * necesite para que el efecto sea medible pero la animación
 * siga siendo fluida. */
#define ITERACIONES_EXTRA 400

typedef struct {
    float x;
    float y;
    float vx;
    float vy;
} Particula;

Particula particulas[NUM_PARTICULAS];

double inicioMedicion;
double tiempoActualizacion = 0.0;
long long actualizaciones = 0;
int medicionTerminada = 0;

/* Contador de partículas que cayeron en la región costosa en el
 * último frame, solo para reportar el nivel de desbalance. */
long long particulasEnRegionUltimoFrame = 0;

/* ---------------------------------------------------------
 * Inicializar particulas
 * --------------------------------------------------------- */
void inicializarParticulas() {
    for (int i = 0; i < NUM_PARTICULAS; i++) {
        particulas[i].x  = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
        particulas[i].y  = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
        particulas[i].vx = ((float) rand() / RAND_MAX) * 0.01f - 0.005f;
        particulas[i].vy = ((float) rand() / RAND_MAX) * 0.01f - 0.005f;
    }
}

/* ---------------------------------------------------------
 * ¿La partícula i está dentro de la región costosa?
 * --------------------------------------------------------- */
static inline int enRegionCostosa(const Particula *p) {
    return (p->x >= REGION_XMIN && p->x <= REGION_XMAX &&
            p->y >= REGION_YMIN && p->y <= REGION_YMAX);
}

/* ---------------------------------------------------------
 * Cálculo adicional que se ejecuta SOLO para las partículas
 * que caen dentro de la región costosa. Es trabajo puramente
 * computacional (sin efecto visual relevante) cuyo único
 * propósito es simular una carga de trabajo desigual entre
 * iteraciones, para provocar desbalance entre hilos.
 * --------------------------------------------------------- */
static inline void calculoCostoso(Particula *p) {
    float acumulado = 0.0f;
    for (int k = 1; k <= ITERACIONES_EXTRA; k++) {
        acumulado += sinf(p->x * k) * cosf(p->y * k);
    }
    /* Se aplica un efecto mínimo para que el compilador no
     * elimine el cálculo por "código muerto" (dead code
     * elimination), pero sin alterar visiblemente el
     * movimiento de la partícula. */
    p->vx += acumulado * 1e-9f;
    p->vy += acumulado * 1e-9f;
}

/* ---------------------------------------------------------
 * CALCULO: actualizar posicion de las particulas
 *
 * IMPORTANTE: se usa "schedule(runtime)", lo que permite
 * cambiar el tipo de planificación SIN recompilar, definiendo
 * la variable de entorno OMP_SCHEDULE antes de ejecutar. Por
 * ejemplo, en bash:
 *
 *   OMP_SCHEDULE=static   ./particulas
 *   OMP_SCHEDULE=dynamic  ./particulas
 *   OMP_SCHEDULE=guided   ./particulas
 *
 * Esto facilita repetir la prueba varias veces con cada
 * planificación, manteniendo el mismo binario, el mismo N y
 * la misma cantidad de frames.
 * --------------------------------------------------------- */
void actualizarParticulas() {
    double inicioActualizacion = omp_get_wtime();
    long long contadorRegion = 0;

    #pragma omp parallel for schedule(runtime) reduction(+:contadorRegion)
    for (int i = 0; i < NUM_PARTICULAS; i++) {

        particulas[i].x += particulas[i].vx;
        particulas[i].y += particulas[i].vy;

        // Si la particula llega al borde horizontal,
        // cambia la direccion de movimiento
        if (particulas[i].x >= 1.0f || particulas[i].x <= -1.0f) {
            particulas[i].vx = -particulas[i].vx;
        }
        // Si la particula llega al borde vertical,
        // cambia la direccion de movimiento
        if (particulas[i].y >= 1.0f || particulas[i].y <= -1.0f) {
            particulas[i].vy = -particulas[i].vy;
        }

        // Desbalance de carga: solo las particulas dentro de
        // la region costosa realizan trabajo adicional.
        if (enRegionCostosa(&particulas[i])) {
            calculoCostoso(&particulas[i]);
            contadorRegion++;
        }
    }

    particulasEnRegionUltimoFrame = contadorRegion;

    /* ------------------------------------------------------
     * NOTA: las versiones explícitas de abajo se dejan solo
     * como referencia/documentación de las tres planificaciones
     * que se deben probar. La planificación real que se ejecuta
     * es la definida por schedule(runtime) arriba, controlada
     * mediante la variable de entorno OMP_SCHEDULE.
     *
     * #pragma omp parallel for schedule(static)
     * #pragma omp parallel for schedule(dynamic)
     * #pragma omp parallel for schedule(guided)
     * ------------------------------------------------------ */

    tiempoActualizacion += omp_get_wtime() - inicioActualizacion;
    actualizaciones++;
}

// ---------------------------------------------------------
// DIBUJADO: OpenGL dibuja las particulas
// ---------------------------------------------------------
void dibujar() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Primero se actualizan las posiciones durante la ventana de medicion.
    if (!medicionTerminada) {
        actualizarParticulas();
    }

    // Dibujar todas las particulas
    glPointSize(3.0f);
    glBegin(GL_POINTS);

    for (int i = 0; i < NUM_PARTICULAS; i++) {
        if (enRegionCostosa(&particulas[i])) {
            // Particulas dentro de la region costosa se pintan
            // de otro color para visualizar el area de interes.
            glColor3f(1.0f, 0.3f, 0.3f);
        } else {
            glColor3f(1.0f, 1.0f, 1.0f);
        }

        glVertex2f(particulas[i].x, particulas[i].y);
    }
    glEnd();

    // Dibujar el contorno de la region costosa como referencia visual
    glColor3f(0.2f, 0.6f, 1.0f);
    glBegin(GL_LINE_LOOP);
        glVertex2f(REGION_XMIN, REGION_YMIN);
        glVertex2f(REGION_XMAX, REGION_YMIN);
        glVertex2f(REGION_XMAX, REGION_YMAX);
        glVertex2f(REGION_XMIN, REGION_YMAX);
    glEnd();

    glutSwapBuffers();

    if (!medicionTerminada && omp_get_wtime() - inicioMedicion >= DURACION_PRUEBA) {
        omp_sched_t tipoPlanificacion;
        int tamanoChunk;
        omp_get_schedule(&tipoPlanificacion, &tamanoChunk);

        const char *nombrePlanificacion = "desconocida";
        switch (tipoPlanificacion) {
            case omp_sched_static:  nombrePlanificacion = "static";  break;
            case omp_sched_dynamic: nombrePlanificacion = "dynamic"; break;
            case omp_sched_guided:  nombrePlanificacion = "guided";  break;
            case omp_sched_auto:    nombrePlanificacion = "auto";    break;
            default: break;
        }

        printf("\nResultados de la prueba (%.0f segundos):\n", DURACION_PRUEBA);
        printf("Planificacion (OMP_SCHEDULE): %s, chunk=%d\n", nombrePlanificacion, tamanoChunk);
        printf("Hilos maximos disponibles: %d\n", omp_get_max_threads());
        printf("Particulas: %d\n", NUM_PARTICULAS);
        printf("Particulas en region costosa (ultimo frame): %lld\n", particulasEnRegionUltimoFrame);
        printf("Actualizaciones (frames medidos): %lld\n", actualizaciones);
        printf("Tiempo acumulado de actualizacion: %.6f segundos\n", tiempoActualizacion);
        printf("Tiempo promedio por actualizacion: %.6f segundos\n",
               tiempoActualizacion / actualizaciones);
        medicionTerminada = 1;
    }
}

// ---------------------------------------------------------
// Temporizador de animacion
// ---------------------------------------------------------
void temporizador(int valor) {

    glutPostRedisplay();
    glutTimerFunc(16, temporizador, 0);
}
// ---------------------------------------------------------
// Configuracion inicial de OpenGL
// ---------------------------------------------------------

void configurarOpenGL() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

// ---------------------------------------------------------
// Programa principal
// ---------------------------------------------------------
int main(int argc, char **argv) {

    // Si el usuario no definio OMP_SCHEDULE, se usa "static" por
    // defecto para que el programa siempre tenga una planificacion
    // valida incluso sin configurar la variable de entorno.
    if (getenv("OMP_SCHEDULE") == NULL) {
        omp_set_schedule(omp_sched_static, 0);
    }

    inicializarParticulas();
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    glutInitWindowSize(800, 600);
    glutCreateWindow("Simulacion de Particulas - Region Costosa");
    configurarOpenGL();

    glutDisplayFunc(dibujar);
    glutTimerFunc(16, temporizador, 0);
    inicioMedicion = omp_get_wtime();
    glutMainLoop();

    return 0;
}
