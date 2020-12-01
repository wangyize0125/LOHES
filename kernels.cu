#include <stdio.h>

extern "C"{
/* -------------------------------------------------------- */
// Name: cal_cv
// TODO: calculate the constraint values of wind turbines 
//       and wave energy converters
/* -------------------------------------------------------- */
__global__ void cal_cv(float* inputs, float* cvs, float* sx, float* sy, int* num, int* nd){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId > (*num)){
        // too much threads
        return;
    }

    // varibale indicating whether it is ok for this layout
    bool is_ok = true;

    float x, y, cv_true = 0, cv_false = 0;
    // loop each devices to check
    for(int i = 0; i < (*nd) - 1; i++){
        for(int j = i + 1; j < (*nd); j++){
            // calculate spacing between them
            x = abs(inputs[threadId * (*nd) * 2 + j] - inputs[threadId * (*nd) * 2 + i]);
            y = abs(inputs[threadId * (*nd) * 2 + j + (*nd)] - inputs[threadId * (*nd) * 2 + i + (*nd)]);

            // // debug
            // if(threadId == 80){
            //     printf("%f %f\n", x, y);
            // }
            
            // check
            if(x > (*sx) || y > (*sy)){
                cv_true -= (x + y);
            }
            else{
                cv_false += (x + y);
                is_ok = false;
            }
        }
    }

    // output
    if(is_ok){
        cvs[threadId] = cv_true;
    }
    else{
        cvs[threadId] = cv_false;
    }
}

/* -------------------------------------------------------- */
// Name: pre_energy
// TODO: calculate the energy outputs of the wind turbines 
/* -------------------------------------------------------- */
__global__ void pre_energy(float* inputs, float* energys, float* direc_addr, float* vel_addr, float* start_vel_addr,
    float* cut_vel_addr, float* prob_addr, int* num_addr, int* nd_addr){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId > (*num_addr)){
        // too much threads
        return;
    }

    float total_energy = 0;

    // accumulate the energy
    energys[threadId] += (*prob_addr) * total_energy;
}
}