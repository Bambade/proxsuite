#ifndef SOLVE_LINSYS_DATA_H
#define SOLVE_LINSYS_DATA_H
//#include "osqp.h"
/* create data and solutions structure */

typedef struct {
c_int test_solve_KKT_n;
c_int test_solve_KKT_m;
csc * test_solve_KKT_A;
csc * test_solve_KKT_Pu;
c_float test_solve_KKT_rho;
c_float test_solve_KKT_sigma;
csc * test_solve_KKT_KKT;
c_float * test_solve_KKT_rhs;
c_float * test_solve_KKT_x;
} solve_linsys_sols_data;

/* function prototypes */
solve_linsys_sols_data *  generate_problem_solve_linsys_sols_data();
void clean_problem_solve_linsys_sols_data(solve_linsys_sols_data * data);


/* function to define problem data */
solve_linsys_sols_data *  generate_problem_solve_linsys_sols_data(){

solve_linsys_sols_data * data = (solve_linsys_sols_data *)c_malloc(sizeof(solve_linsys_sols_data));

data->test_solve_KKT_n = 3;
data->test_solve_KKT_m = 4;

// Matrix test_solve_KKT_A
//------------------------
data->test_solve_KKT_A = (csc*) c_malloc(sizeof(csc));
data->test_solve_KKT_A->m = 4;
data->test_solve_KKT_A->n = 3;
data->test_solve_KKT_A->nz = -1;
data->test_solve_KKT_A->nzmax = 5;
data->test_solve_KKT_A->x = (c_float*) c_malloc(5 * sizeof(c_float));
data->test_solve_KKT_A->x[0] = 0.42278467327012780874;
data->test_solve_KKT_A->x[1] = 0.63318439927411640511;
data->test_solve_KKT_A->x[2] = 0.96743595249367664302;
data->test_solve_KKT_A->x[3] = 0.43263079080478716865;
data->test_solve_KKT_A->x[4] = 0.66929729857452024966;
data->test_solve_KKT_A->i = (c_int*) c_malloc(5 * sizeof(c_int));
data->test_solve_KKT_A->i[0] = 0;
data->test_solve_KKT_A->i[1] = 2;
data->test_solve_KKT_A->i[2] = 1;
data->test_solve_KKT_A->i[3] = 3;
data->test_solve_KKT_A->i[4] = 2;
data->test_solve_KKT_A->p = (c_int*) c_malloc((3 + 1) * sizeof(c_int));
data->test_solve_KKT_A->p[0] = 0;
data->test_solve_KKT_A->p[1] = 2;
data->test_solve_KKT_A->p[2] = 4;
data->test_solve_KKT_A->p[3] = 5;


// Matrix test_solve_KKT_Pu
//-------------------------
data->test_solve_KKT_Pu = (csc*) c_malloc(sizeof(csc));
data->test_solve_KKT_Pu->m = 3;
data->test_solve_KKT_Pu->n = 3;
data->test_solve_KKT_Pu->nz = -1;
data->test_solve_KKT_Pu->nzmax = 6;
data->test_solve_KKT_Pu->x = (c_float*) c_malloc(6 * sizeof(c_float));
data->test_solve_KKT_Pu->x[0] = 0.03530681337232168676;
data->test_solve_KKT_Pu->x[1] = 0.01036211046843158422;
data->test_solve_KKT_Pu->x[2] = 0.00304115050621230325;
data->test_solve_KKT_Pu->x[3] = 0.11275953295680966881;
data->test_solve_KKT_Pu->x[4] = 0.03309352006780613004;
data->test_solve_KKT_Pu->x[5] = 0.89092108249253432195;
data->test_solve_KKT_Pu->i = (c_int*) c_malloc(6 * sizeof(c_int));
data->test_solve_KKT_Pu->i[0] = 0;
data->test_solve_KKT_Pu->i[1] = 0;
data->test_solve_KKT_Pu->i[2] = 1;
data->test_solve_KKT_Pu->i[3] = 0;
data->test_solve_KKT_Pu->i[4] = 1;
data->test_solve_KKT_Pu->i[5] = 2;
data->test_solve_KKT_Pu->p = (c_int*) c_malloc((3 + 1) * sizeof(c_int));
data->test_solve_KKT_Pu->p[0] = 0;
data->test_solve_KKT_Pu->p[1] = 1;
data->test_solve_KKT_Pu->p[2] = 3;
data->test_solve_KKT_Pu->p[3] = 6;

data->test_solve_KKT_rho = 4.00000000000000000000;
data->test_solve_KKT_sigma = 1.00000000000000000000;

// Matrix test_solve_KKT_KKT
//--------------------------
data->test_solve_KKT_KKT = (csc*) c_malloc(sizeof(csc));
data->test_solve_KKT_KKT->m = 7;
data->test_solve_KKT_KKT->n = 7;
data->test_solve_KKT_KKT->nz = -1;
data->test_solve_KKT_KKT->nzmax = 23;
data->test_solve_KKT_KKT->x = (c_float*) c_malloc(23 * sizeof(c_float));
data->test_solve_KKT_KKT->x[0] = 1.03530681337232177697;
data->test_solve_KKT_KKT->x[1] = 0.01036211046843158422;
data->test_solve_KKT_KKT->x[2] = 0.11275953295680966881;
data->test_solve_KKT_KKT->x[3] = 0.42278467327012780874;
data->test_solve_KKT_KKT->x[4] = 0.63318439927411640511;
data->test_solve_KKT_KKT->x[5] = 0.01036211046843158422;
data->test_solve_KKT_KKT->x[6] = 1.00304115050621223126;
data->test_solve_KKT_KKT->x[7] = 0.03309352006780613004;
data->test_solve_KKT_KKT->x[8] = 0.96743595249367664302;
data->test_solve_KKT_KKT->x[9] = 0.43263079080478716865;
data->test_solve_KKT_KKT->x[10] = 0.11275953295680966881;
data->test_solve_KKT_KKT->x[11] = 0.03309352006780613004;
data->test_solve_KKT_KKT->x[12] = 1.89092108249253421093;
data->test_solve_KKT_KKT->x[13] = 0.66929729857452024966;
data->test_solve_KKT_KKT->x[14] = 0.42278467327012780874;
data->test_solve_KKT_KKT->x[15] = -0.25000000000000000000;
data->test_solve_KKT_KKT->x[16] = 0.96743595249367664302;
data->test_solve_KKT_KKT->x[17] = -0.25000000000000000000;
data->test_solve_KKT_KKT->x[18] = 0.63318439927411640511;
data->test_solve_KKT_KKT->x[19] = 0.66929729857452024966;
data->test_solve_KKT_KKT->x[20] = -0.25000000000000000000;
data->test_solve_KKT_KKT->x[21] = 0.43263079080478716865;
data->test_solve_KKT_KKT->x[22] = -0.25000000000000000000;
data->test_solve_KKT_KKT->i = (c_int*) c_malloc(23 * sizeof(c_int));
data->test_solve_KKT_KKT->i[0] = 0;
data->test_solve_KKT_KKT->i[1] = 1;
data->test_solve_KKT_KKT->i[2] = 2;
data->test_solve_KKT_KKT->i[3] = 3;
data->test_solve_KKT_KKT->i[4] = 5;
data->test_solve_KKT_KKT->i[5] = 0;
data->test_solve_KKT_KKT->i[6] = 1;
data->test_solve_KKT_KKT->i[7] = 2;
data->test_solve_KKT_KKT->i[8] = 4;
data->test_solve_KKT_KKT->i[9] = 6;
data->test_solve_KKT_KKT->i[10] = 0;
data->test_solve_KKT_KKT->i[11] = 1;
data->test_solve_KKT_KKT->i[12] = 2;
data->test_solve_KKT_KKT->i[13] = 5;
data->test_solve_KKT_KKT->i[14] = 0;
data->test_solve_KKT_KKT->i[15] = 3;
data->test_solve_KKT_KKT->i[16] = 1;
data->test_solve_KKT_KKT->i[17] = 4;
data->test_solve_KKT_KKT->i[18] = 0;
data->test_solve_KKT_KKT->i[19] = 2;
data->test_solve_KKT_KKT->i[20] = 5;
data->test_solve_KKT_KKT->i[21] = 1;
data->test_solve_KKT_KKT->i[22] = 6;
data->test_solve_KKT_KKT->p = (c_int*) c_malloc((7 + 1) * sizeof(c_int));
data->test_solve_KKT_KKT->p[0] = 0;
data->test_solve_KKT_KKT->p[1] = 5;
data->test_solve_KKT_KKT->p[2] = 10;
data->test_solve_KKT_KKT->p[3] = 14;
data->test_solve_KKT_KKT->p[4] = 16;
data->test_solve_KKT_KKT->p[5] = 18;
data->test_solve_KKT_KKT->p[6] = 21;
data->test_solve_KKT_KKT->p[7] = 23;

data->test_solve_KKT_rhs = (c_float*) c_malloc(7 * sizeof(c_float));
data->test_solve_KKT_rhs[0] = -0.60718569987063708560;
data->test_solve_KKT_rhs[1] = 0.12682784711186986804;
data->test_solve_KKT_rhs[2] = -0.89227404342979033114;
data->test_solve_KKT_rhs[3] = 0.84146497237014306059;
data->test_solve_KKT_rhs[4] = 0.18803508698068596705;
data->test_solve_KKT_rhs[5] = 0.33057100813532613870;
data->test_solve_KKT_rhs[6] = 0.41050391297026284088;
data->test_solve_KKT_x = (c_float*) c_malloc(7 * sizeof(c_float));
data->test_solve_KKT_x[0] = 0.67236089919091335254;
data->test_solve_KKT_x[1] = 0.28550326844509271718;
data->test_solve_KKT_x[2] = -0.33461127649968114284;
data->test_solve_KKT_x[3] = 0.28426388308403949257;
data->test_solve_KKT_x[4] = 0.27620612644823616666;
data->test_solve_KKT_x[5] = 0.20177400861579478097;
data->test_solve_KKT_x[6] = 0.12351750480475187643;

return data;

}

/* function to clean data struct */
void clean_problem_solve_linsys_sols_data(solve_linsys_sols_data * data){

c_free(data->test_solve_KKT_A->x);
c_free(data->test_solve_KKT_A->i);
c_free(data->test_solve_KKT_A->p);
c_free(data->test_solve_KKT_A);
c_free(data->test_solve_KKT_Pu->x);
c_free(data->test_solve_KKT_Pu->i);
c_free(data->test_solve_KKT_Pu->p);
c_free(data->test_solve_KKT_Pu);
c_free(data->test_solve_KKT_KKT->x);
c_free(data->test_solve_KKT_KKT->i);
c_free(data->test_solve_KKT_KKT->p);
c_free(data->test_solve_KKT_KKT);
c_free(data->test_solve_KKT_rhs);
c_free(data->test_solve_KKT_x);

c_free(data);

}

#endif
