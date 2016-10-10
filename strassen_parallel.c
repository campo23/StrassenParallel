#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkl.h"
#include <mpi.h>

#define END -1

//funzione che alloca una matrice
float **matrix_alloc(int rows, int columns) {
	float **mx;
	mx = (float **) malloc (rows * sizeof(float*));
	int i;	
	for(i=0; i<rows; i++) 
		mx[i] = (float *)mkl_malloc( columns*sizeof( float ), 64 );
	return (mx);
}

//funzione che dealloca una matrice
float **matrix_dealloc(float **mx, int rows) {
	int i;
	for(i = 0; i < rows; i++) {
		mkl_free(mx[i]);
	}
	free(mx);
	return (NULL);
}

int main(int argc, char *argv[]) {

	int i, j, rank, size;

	MPI_Status status;
	MPI_Request rqst;
	MPI_Init (&argc, &argv);
	MPI_Comm comm1;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

   	if(size <= 1) {
		printf("ERROR! not enough CPUs!\n");
		return 0;
	}

    double s_time, s1_time; 
	int dim = 8192;
	//dimensione degli array che contengono le 4 sottomatrici di A e le 4 di B
	int dimP = (dim/2)*(dim/2);

	if(rank == 0) {
		//le due matrici sono contenute in un unico array che ha dimensione dim*dim (A) + dim*dim (B)
		int dim2 = (dim*dim)*2;

		//mkl_alloc è la funzione della libreria mkl.h per allocare gli array
		float *AB = (float *)mkl_malloc( dim2*sizeof( float ), 64 ), *C;

		for(i = 0; i<dim2; i++) AB[i] = 1;

		float **send, **recvC;

		send = matrix_alloc(7, dimP*2);

		s1_time = MPI_Wtime();	

		//calcolo le matrici da inviare agli slave

		//M1 = (A11+A22)x(B11+B22)
		for(i = 0; i < dimP; i++) {
			send[0][i] = AB[i] + AB[i + (dimP*3)];
		}
		for(i = 0; i < dimP; i++) {
			send[0][i+dimP] = AB[i + (dim*dim)] + AB[i + (dim*dim) + (dimP*3)];
		}
		//M2 = (A21+A22)xB11
		for(i = 0; i < dimP; i++) {
			send[1][i] = AB[i + (dimP*2)] + AB[i + (dimP*3)];
		}
		for(i = 0; i < dimP; i++) {
			send[1][i+dimP] = AB[i + (dim*dim)];
		}
		//M3 = A11x(B12-B22)
		for(i = 0; i < dimP; i++) {
			send[2][i] = AB[i];
		}
		for(i = 0; i < dimP; i++) {
			send[2][i+dimP] = AB[i + (dim*dim) + dimP] - AB[i + (dim*dim) + (dimP*3)];
		}
		//M4 = A22x(B21-B11)
		for(i = 0; i < dimP; i++) {
			send[3][i] = AB[i + (dimP*3)];
		}
		for(i = 0; i < dimP; i++) {
			send[3][i+dimP] = AB[i + (dim*dim) + (dimP*2)] - AB[i + (dim*dim)];
		}
		//M5 = (A11+A12)xB22
		for(i = 0; i < dimP; i++) {
			send[4][i] = AB[i] + AB[i + dimP];
		}
		for(i = 0; i < dimP; i++) {
			send[4][i+dimP] = AB[i + (dim*dim) + (dimP*3)];
		}
		//M6 = (A21-A11)x(B11+B12)
		for(i = 0; i < dimP; i++) {
			send[5][i] = AB[i + dimP*2] - AB[i];
		}
		for(i = 0; i < dimP; i++) {
			send[5][i+dimP] = AB[i + (dim*dim)] + AB[i +(dim*dim) + dimP];
		}
		//M7 = (A12-A22)x(B21+B22)
		for(i = 0; i < dimP; i++) {
			send[6][i] = AB[i + dimP] - AB[i + (dimP*3)];
		}
		for(i = 0; i < dimP; i++) {
			send[6][i+dimP] = AB[i + (dim*dim) + (dimP*2)] + AB[i + (dim*dim) + (dimP*3)];
		}

		mkl_free(AB);

		int stop = 0;
		int treelevel = 2;
		
		//miss indica il numero di send totali da fare, equivalente al numero di M
		int miss = 7;

		//miss2 conta quante send vengono effettuate a ogni ciclo, in modo da sapere
		//esattamente quante receive fare dopo
		int miss2;

		int s = 0, t = 0;

		recvC = matrix_alloc(7, dimP);

		s_time=MPI_Wtime();
		double boh_time;
		while(miss!=0) {
			miss2  = 0;
			for(i=1; i<=7; i++) {
				if(i<size && miss!=0) {
					miss--;
					miss2++;
					MPI_Send(&send[s][0], dimP*2, MPI_FLOAT, i, treelevel, MPI_COMM_WORLD);
					mkl_free(send[s]);
					s++;
				}
				else break;
			}
			boh_time = MPI_Wtime();

			for(i=1; i<=miss2; i++) {
				if(i<size) {
					MPI_Recv(&recvC[t][0], dimP, MPI_FLOAT, i, treelevel, MPI_COMM_WORLD, &status);
					t++;
				}
				else break;
			}
			
		}

		for(i = 1; i <= 7; i++) {
			if(i<size)
				MPI_Send(&stop, 1, MPI_INT, i, END, MPI_COMM_WORLD);
		}

		free(send);

		C = (float *)mkl_malloc( dim*dim*sizeof( float ), 64 );

		//ricompongo C
		//C11 = M1 + M4 - M5 + M7
		for(i = 0; i < dimP; i++) {
			C[i] = recvC[0][i] + recvC[3][i] - recvC[4][i] + recvC[6][i];
		}

		//C12 = M3 + M5
		for(i = 0; i < dimP; i++) {
			C[i+dimP] = recvC[2][i] + recvC[4][i];
		}

		//C21 = M2 + M4
		for(i = 0; i < dimP; i++) {
			C[i+dimP*2] = recvC[1][i] - recvC[3][i];
		}

		//C22 = M1 + M3 - M2 + M6
		for(i = 0; i < dimP; i++) {
			C[i+dimP*3] = recvC[0][i] + recvC[2][i] - recvC[1][i] + recvC[5][i];
		}

		/*for(i=dim*dim-10; i<dim*dim; i++) {
			printf("%f ", C[i]);
		}*/

		printf("rank %d calculation time : %f\n", rank, MPI_Wtime() - s_time);

		matrix_dealloc(recvC, 7);
		mkl_free(C);
	}

	else {
		float *recvA, *C;
		float alpha, beta;
		alpha = 1.0; beta = 0.0;
		int  m, treelevel, father, stop;
		double s_time2;

		while(1) {

			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			
			treelevel = status.MPI_TAG;
			father = status.MPI_SOURCE;

			if(treelevel == -1) {
				MPI_Recv(&stop, 1, MPI_INT, father, treelevel, MPI_COMM_WORLD, &status);
				break;
			}

			int dimT = dim/(pow(2, treelevel-1)) * dim/(pow(2, treelevel-1));

			recvA = (float *)mkl_malloc( dimT*2*sizeof( float ), 64 );

			//ricevo A e B
			MPI_Recv(&recvA[0], dimT*2, MPI_FLOAT, father, treelevel, MPI_COMM_WORLD, &status);

			//conto fino a che rank arrivano i nodi nel livello del nodo corrente
			//nel primo livello i nodi arrivano fino a rank 7, nel secondo fino a rank 7+7^2, 
			//nel terzo 7+7^2+7^3 e cosi via
			int level_size = 0;

			for(i=1; i<treelevel-1; i++) level_size += pow(7, i);

			s_time = MPI_Wtime();

			//controllo se il nodo corrente ha figli
			if(size > level_size && (rank+pow(7, treelevel-1))<size) {
				
				float **send, **recvC;

				send = matrix_alloc(7, dimT/2);

				//M1
				for(i = 0; i < dimT/4; i++) {
					send[0][i] = recvA[i] + recvA[i + dimT/4 + dimT/2];
				}
				for(i = 0; i < dimT/4; i++) {
					send[0][i + (dimT/4)] = recvA[i+dimT] + recvA[i + dimT + dimT/4 + dimT/2];
				}
				//M2
				for(i = 0; i < dimT/4; i++) {
					send[1][i] = recvA[i + dimT/2] + recvA[i + dimT/4 + dimT/2];
				}
				for(i = 0; i < dimT/4; i++) {
					send[1][i + (dimT/4)] = recvA[i + dimT];
				}
				//M3
				for(i = 0; i < dimT/4; i++) {
					send[2][i] = recvA[i];
				}
				for(i = 0; i < dimT/4; i++) {
					send[2][i + (dimT/4)] = recvA[i + +dimT + dimT/4] - recvA[i + dimT + dimT/4 + dimT/2];
				}
				//M4
				for(i = 0; i < dimT/4; i++) {
					send[3][i] = recvA[i + + dimT/4 + dimT/2];
				}
				for(i = 0; i < dimT/4; i++) {
					send[3][i + (dimT/4)] = recvA[i + dimT + dimT/2] - recvA[i + dimT];
				}
				//M5
				for(i = 0; i < dimT/4; i++) {
					send[4][i] = recvA[i] + recvA[i + dimT/4];
				}
				for(i = 0; i < dimT/4; i++) {
					send[4][i + (dimT/4)] = recvA[i + dimT + dimT/4 + dimT/2];
				}
				//M6
				for(i = 0; i < dimT/4; i++) {
					send[5][i] = recvA[i + dimT/2] - recvA[i];
				}
				for(i = 0; i < dimT/4; i++) {
					send[5][i + (dimT/4)] = recvA[i + dimT] + recvA[i + dimT + dimT/4];
				}
				//M7
				for(i = 0; i < dimT/4; i++) {
					send[6][i] = recvA[i + dimT/4] - recvA[i + dimT/4 + dimT/2];
				}
				for(i = 0; i < dimT/4; i++) {
					send[6][i + (dimT/4)] = recvA[i + dimT + dimT/2] + recvA[i + dimT + dimT/4 + dimT/2];
				}

				mkl_free(recvA);

				recvC = matrix_alloc(7, dimT/4);
				int miss = 7;
				int miss2;
				int s = 0, t = 0;
				while(miss!=0) {
					miss2  = 0;
					for(i=1; i<=7; i++) {
						if(rank+(i*pow(7, treelevel-1))<size && miss!=0) {
							miss--;
							miss2++;
							MPI_Send(&send[s][0], dimT/2, MPI_FLOAT, rank+(i*7), treelevel+1, MPI_COMM_WORLD);
							mkl_free(send[s]);
							s++;
						}
						else break;
					}
					for(i=1; i<=miss2; i++) {
						if(rank+(i*7)<size) {
							MPI_Recv(&recvC[t][0], dimT/2, MPI_FLOAT, rank+(i*7), treelevel+1, MPI_COMM_WORLD, &status);
							t++;
						}
						else break;
					}
				}

				free(send);

				stop = 0;

				for(i = 1; i <= 7; i++) {
					if(rank+(i*7)<size)
						MPI_Send(&stop, 1, MPI_INT, rank+(i*7), END, MPI_COMM_WORLD);
				}

				C = (float *)mkl_malloc( dimT*sizeof( float ), 64 );

				//ricompongo C
				//C11
				for(i = 0; i < dimT/4; i++) {
					C[i] = recvC[0][i] + recvC[3][i] - recvC[4][i] + recvC[6][i];
				}

				//C12
				for(i = 0; i < dimT/4; i++) {
					C[i+dimT/4] = recvC[2][i] + recvC[4][i];
				}

				//C21
				for(i = 0; i < dimT/4; i++) {
					C[i+dimT/2] = recvC[1][i] - recvC[3][i];
				}

				//C22
				for(i = 0; i < dimT/4; i++) {
					C[i + dimT/4 + dimT/2] = recvC[0][i] + recvC[2][i] - recvC[1][i] + recvC[5][i];
				}

				//invio C al father
				MPI_Send(&C[0], dimT, MPI_FLOAT, father, treelevel, MPI_COMM_WORLD);

				s_time = MPI_Wtime() - s_time;
				s_time2 += s_time;

				//libero la memoria
				mkl_free(C);
				recvC = matrix_dealloc(recvC, 7);
			}

			//se il nodo corrente non ha figli, eseguo direttamente la moltiplicazione tra le due matrici ricevute
			else{

				//divido recvA in due array A1 e B1 da poter passare a cblas_dgemm

				double *A1 = (double *)mkl_malloc( dimT*sizeof( double ), 64 );
				double *B1 = (double *)mkl_malloc( dimT*sizeof( double ), 64 );

				for(i = 0; i < dimT; i++) {
					A1[i] = (double)recvA[i];
				}

				for(i = 0; i < dimT; i++) {
					B1[i] = (double)recvA[i + dimT];
				}

				mkl_free(recvA);
				
				m = sqrt(dimT);

				double *C1 = (double *)mkl_malloc( dimT*sizeof( double ), 64 );

				//eseguo la routine dgemm su A1 e B1s
		  		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, alpha, A1, m, B1, m, beta, C1, m);

		  		C = (float *)mkl_malloc( dimT*sizeof( float ), 64 );

		  		for(i = 0; i < dimT; i++) {
					C[i] = (float)C1[i];
				}
		 		
				MPI_Send(&C[0], dimT, MPI_FLOAT, father, treelevel, MPI_COMM_WORLD);	

				s_time = MPI_Wtime() - s_time;
				s_time2 += s_time;

				//libero la memoria
				
		  		mkl_free(A1);
		  		mkl_free(B1);
		  		mkl_free(C);
		  		mkl_free(C1);
	  		}
		
		}
		printf("rank %d calculation time : %f\n", rank, s_time2);
	}

	MPI_Finalize();
}