#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
#include "mpi.h"

#define MAXNAME 250
#define MAXL 250
#define MAXW 650
#define LONGS 5000

// Functions
int parse_string(char* s, char* a[]);
void char_to_float(char* in[], double out[], int nw);

/* Program for computing velocity autocorrelation 
 * function from a LAMMPS dump file
 * version with MPI parallelization
 *
 * Syntax: compute-vacf <dumpfile> <outputfile>
 * Notes: - <dumpfile> is the standard LAMMPS .d file
 *          and <outputfile> is the name of the file where
 *          the output is supposed to be written
 *        - <dumpfile> and <outputfile> have to be 
 *          specified with their full paths
	      - Runs for arbitrary number of processors,
			that is set in the submission script
 *        - Number of atoms assumed fixed throught 
 *          the computation.
          - Structure of the file - order and names of 
            variables in the dumpfile - is hardcoded.
            To change, need to modify the code.
          - Outputs a file with Time step | vacf_x | vacf_y | vacf_z
          - Array sizes above (#define ...) may have to be 
            made larger based on the input file
   Last modified: February 10 2017      
*/
            
int main(int argc, char *argv[])
{
    // Miscelaneous variables
    char *file_in, *file_out;
    char *sp;
    char s[MAXL];
    int frame=0, nat, k, itm, ktm, jtm, k2;
    int iat,nw,ind,tmp;
	int ptot, my_rank;
    char *wa[MAXW];
    double w[MAXW];
    MPI_Status status;   
 
    // Pointers to files
    FILE *fpi, *fpo;
   
	// MPI initialization (so the parameter list is 
	// like in serial; this is apparently not 
	// guaranteed by the standard so if there is 
	// trouble with args checked later this would 
	// be it
    MPI_Init(&argc, &argv);
	// Check the total number of processors
    MPI_Comm_size(MPI_COMM_WORLD, &ptot);
	// Check the rank of current processor
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   	// Usage check
    if(argc!=3){
        printf("Error\n");
        printf("Syntax compute_vacf <dumpfile> <outputfile>\n");
        exit(1);
    }
   
    // Get the file names
    file_in=argv[1];
    file_out=argv[2];
   
	// Read and broadcast initial information on processor 0
    // ----------------------------------------------------------
    // #frames and #atoms
    if (my_rank == 0){	
        fpi=fopen(file_in,"r");
        while((sp = fgets(s,MAXL,fpi))!= NULL){
            if(!strcmp(sp,"ITEM: TIMESTEP\n"))
                frame++;
            if(!strcmp(sp,"ITEM: NUMBER OF ATOMS\n"))
                nat = atoi(fgets(s,MAXL,fpi));  
        }
        printf("Number of frames: %d and number of atoms %d\n",frame, nat);
        fflush(stdout);
        rewind(fpi);
    }
    MPI_Bcast(&frame, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nat, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Time step array
    double* time=(double*) malloc(frame*sizeof(double));
    if (my_rank == 0){	
        k=0;
        while((sp = fgets(s,MAXL,fpi))!= NULL){
            if(!strcmp(sp,"ITEM: TIMESTEP\n")){
                sp = fgets(s,MAXL,fpi);
                nw=parse_string(s,wa);
                char_to_float(wa,w,nw);
                time[k]=w[0];
                k++;
            }
        }
        fclose(fpi);
    }
    MPI_Bcast(time, frame, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Allocate and initialize arrays
    // ----------------------------------------------------------
    // Total VACF
    double** VACF= (double**) malloc(3*sizeof(double*));
    for(k=0;k<3;k++)
        VACF[k]= (double*) malloc(frame*sizeof(double));
    for(k=0;k<frame;k++)
        for(k2=0;k2<3;k2++)
            VACF[k2][k]=0.0;
    // Local total VACF
    double** local_VACF= (double**) malloc(3*sizeof(double*));
    for(k=0;k<3;k++)
        local_VACF[k]= (double*) malloc(frame*sizeof(double));
    for(k=0;k<frame;k++)
        for(k2=0;k2<3;k2++)
            local_VACF[k2][k]=0.0;
    // Temporary velocity storage for every atom
    double** vel_temp= (double**) malloc(frame*sizeof(double*));
    for(k=0;k<frame;k++)
        vel_temp[k]= (double*) malloc(3*sizeof(double));
    // Temporary VACF for an atom
    double* Vtemp=(double*) malloc(3*sizeof(double));

    // Open the datafile - all read from the same file
    fpi=fopen(file_in,"r");
    
    // Main computation
   	// ----------------------------------------------------------
    // Loop for each atom, store velocity components 
    // from each frame, than proceed with computations
    // for that one atom
    // 
    // Define bounds for each processor (last one will deal with any
    // irregularities)
    const int del_k = nat/ptot; 
    const int kp0 = my_rank*del_k + 1;
    int kpN = (my_rank+1)*del_k;
    //
    for(ktm=kp0;ktm<=kpN;ktm++){
        if (my_rank==ptot-1)
            kpN=nat;    
        tmp=0;
        // Track progress
        if(ktm%100==0){
            printf("Processor %d, atom #=%d \n", my_rank, ktm);
            fflush(stdout);
        }            
        while((sp = fgets(s,MAXL,fpi))!= NULL){
            // Change the header here if needed
            if(!strcmp(sp,"ITEM: ATOMS id x y z vx vy vz c_myPE c_myKE c_myStress[1] c_myStress[2] c_myStress[3] c_myStress[4] c_myStress[5] c_myStress[6] \n")){
                iat=0;
                // Get the data as a string, parse it into words and
                // then convert each word to a float 
                while(iat<nat){
                    sp = fgets(s,MAXL,fpi);
                    nw=parse_string(s,wa);
                    char_to_float(wa,w,nw);
                    ind=w[0]-1;
                    // If it is the current atom ktm 
                    // save the velocity data, break 
                    // the inner while loop and move on 
                    // to the next frame
                    if(ind+1==ktm){
                        vel_temp[tmp][0]=w[4];
                        vel_temp[tmp][1]=w[5];                            
                        vel_temp[tmp][2]=w[6];
                        tmp++;
                        break;
                    }
                    iat++;
                }
            }
        }
        // Per atom computation
        for(itm=0;itm<frame;itm++){
            // Reinitialize the sum   
            Vtemp[0]=0.0,Vtemp[1]=0.0,Vtemp[2]=0.0;
            for(jtm=0;jtm<=frame-itm-1;jtm++){
                Vtemp[0]=Vtemp[0]+vel_temp[jtm][0]*vel_temp[jtm+itm][0];
                Vtemp[1]=Vtemp[1]+vel_temp[jtm][1]*vel_temp[jtm+itm][1];
                Vtemp[2]=Vtemp[2]+vel_temp[jtm][2]*vel_temp[jtm+itm][2];
            }
            // Add to the VACF matrix
            local_VACF[0][itm]=local_VACF[0][itm]+Vtemp[0]/(frame-itm);
            local_VACF[1][itm]=local_VACF[1][itm]+Vtemp[1]/(frame-itm);
            local_VACF[2][itm]=local_VACF[2][itm]+Vtemp[2]/(frame-itm);
        }
        rewind(fpi);
    }

    // Collect and add all the local VACF data on processor 0
   	// ----------------------------------------------------------
    MPI_Reduce(&(local_VACF[0][0]), &(VACF[0][0]), frame, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&(local_VACF[1][0]), &(VACF[1][0]), frame, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&(local_VACF[2][0]), &(VACF[2][0]), frame, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Correct for the number of atoms and write the result 
    // ----------------------------------------------------------
    if (my_rank == 0){
       // Correct
       for(k=0;k<frame;k++)
            for(k2=0;k2<3;k2++)
                VACF[k2][k]=VACF[k2][k]/nat;
        // Write the data
        fpo=fopen(file_out,"w");
        for(jtm=0;jtm<frame;jtm++){
            fprintf(fpo, "%f %f %f %f\n", time[jtm], VACF[0][jtm],VACF[1][jtm],VACF[2][jtm]);
        }
        fclose(fpo);
    }

    // Free arrays
    // ----------------------------------------------------------
    for(k=0;k<3;k++)
        free(VACF[k]);
    free(VACF);
    for(k=0;k<3;k++)
        free(local_VACF[k]);
    free(local_VACF);
    for(k=0;k<frame;k++)
        free(vel_temp[k]);
    free(vel_temp);
    free(Vtemp);
    free(time);
    
    // Close the input file
	// ----------------------------------------------------------
    fclose(fpi);

    // Close MPI
    MPI_Finalize();
}

/* Function to parse a line of input into an aray of words */
/* s - string to be parsed
 * a - string with parsed elements */
int parse_string(char* s,char* a[])
{
    int nw,j; 
    a[0] = strtok(s," \t\n\r\v\f"); 
    nw = 1;				 
    while((a[nw]= strtok(NULL," \t\n\r\v\f"))!=NULL)
        nw++;
   return nw;
}

/* Function to convert array of words to array
 * of doubles
 * in[] - string with pointers to words
 * out[] - string with doubles */ 
void char_to_float(char* in[], double out[], int nw)
{
    int k;
    for(k=0;k<nw;k++)
        out[k]=atof(in[k]);
}


