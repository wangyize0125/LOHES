#include <stdio.h>

extern "C"{
// global variables
// ratio between radian and angle
__device__ const float pi = 3.1415926535;
__device__ const float pi_ratio = pi / 180;
// number of individuals
__device__ const int num_inds = ${num_inds};
// number of devices
__device__ const int num_turbs = ${num_turbs};
// wind velocities
__device__ float wind_vels[num_inds][num_turbs];
// turbulence intensities
__device__ float turb_ints[num_inds][num_turbs];

/* -------------------------------------------------------- */
// Name: cal_cv
// TODO: calculate the constraint values
/* -------------------------------------------------------- */
__device__ inline float cal_cv(float* inputs, float sx, float sy, int num_devs){
    // varibale indicating whether it is ok for this layout
    bool is_ok = true;

    float x, y, cv_true = 0, cv_false = 0;
    // loop each devices to check
    for(int i = 0; i < num_devs - 1; i++){
        for(int j = i + 1; j < num_devs; j++){
            // calculate spacing between them
            x = abs(inputs[j] - inputs[i]);
            y = abs(inputs[j + num_devs] - inputs[i + num_devs]);

            // // debug
            // if(threadId == 80){
            //     printf("%f %f\n", x, y);
            // }
            
            // check
            if(x > sx || y > sy){
                cv_true -= (x + y);
            }
            else{
                cv_false += (x + y);
                is_ok = false;
            }
        }
    }

    // use 1/* to ensure they are compact
    return is_ok? 1 / cv_true: 1 / cv_false;
}

/* -------------------------------------------------------- */
// Name: cal_cv_turb
// TODO: calculate the constraint values of wind turbines
/* -------------------------------------------------------- */
__global__ void cal_cv_turb(float* inputs, float* cvs, float* sx, float* sy){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId >= num_inds){
        // too much threads
        return;
    }

    cvs[threadId] = cal_cv(inputs + threadId * num_turbs * 2, *sx, *sy, num_turbs);
}

/* -------------------------------------------------------- */
// Name: rotate_and_order
// TODO: rotate the wind turbines to face the incoming winds
//       and order them according to x coordinate
/* -------------------------------------------------------- */
__device__ inline void rotate_and_order(float* inputs, float direc, float radius){
    // rotate the wind turbines to face the incoming winds
    // assuming that x axis along the incoming wind direction
    float angle = 0;    // angle between x axis
    float dist = 0;
    for(int i = 0; i < num_turbs; i++){
        // calculate the angle
        angle = atan(inputs[num_turbs + i] / inputs[i]);
        // rotate the coordinate system
        angle += direc * pi_ratio;
        // calculate the distance
        dist = pow(pow(inputs[num_turbs + i], 2) + pow(inputs[i], 2), 0.5);
        // calculate the new coordinates
        // NOTE: here x coordinate are normalized by rotor radius owing to that x/D 
        // is used in the analytical models
        inputs[i] = dist * cos(angle) / radius; inputs[num_turbs + i] = dist * sin(angle);
    }

    // sort the wind turbines in ascending order of x
    for(int i = 0; i < num_turbs - 1; i++){
        for(int j = 0; j < num_turbs - i - 1; j++){
            if(inputs[j] > inputs[j + 1]){
                // to reduce the memory cost of this function
                // use angle and dist as temp variables here
                angle = inputs[j]; dist = inputs[j + num_turbs]; 
                inputs[j] = inputs[j + 1]; inputs[j + num_turbs] = inputs[j + 1 + num_turbs];
                inputs[j + 1] = angle; inputs[j + 1 + num_turbs] = dist;
            }
        }
    }
}

/* -------------------------------------------------------- */
// Name: pre_energy_turb
// TODO: calculate the energy outputs of the wind turbines 
/* -------------------------------------------------------- */
__global__ void pre_energy_turb(float* inputs, float* energys, float* direc_addr, float* vel_addr, float* start_vel_addr,
    float* cut_vel_addr, float* turb_int_addr, float* prob_addr, float* rad_addr, float* cp_addr){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId >= num_inds){
        // too much threads
        return;
    }

    // owing to that the wind velocity and turblence intensity arrays are too large
    // declare them in global memory as can be found at the beginning of this file
    // though the computational time would be lengthened, the developing time reduces
    // here, set the initial wind velocity and turbulence intensity
    for(int i = 0; i < num_turbs; i++){
        wind_vels[threadId][i] = *vel_addr; turb_ints[threadId][i] = *turb_int_addr;
    }

    // // debug
    // if(threadId == 1){
    //     printf("Before ordering: %f\n", *direc_addr);
    //     for(int i = 0; i < num_turbs; i++){
    //         printf("%f, %f\n", inputs[threadId * num_turbs * 2 + i], inputs[threadId * num_turbs * 2 + i + num_turbs]);
    //     }
    // }

    // rotate and order the wind turbines
    rotate_and_order(inputs + threadId * num_turbs * 2, *direc_addr, *rad_addr);

    // // debug
    // if(threadId == 1){
    //     printf("After ordering:\n");
    //     for(int i = 0; i < num_turbs; i++){
    //         printf("%f, %f\n", inputs[threadId * num_turbs * 2 + i], inputs[threadId * num_turbs * 2 + i + num_turbs]);
    //     }
    // }
    
    // re-locate inputs to the wind turbines that are handled in this thread
    inputs += threadId * num_turbs * 2;
    // then, inputs can be utilized as the first wind turbine for each thread

    // parameters of the analytical models
    float ct, kk, eps, a, b, c, d, e, f, k1, k2, delta, x_d, r;
    // loop for each turbine to calculate their wind velocity and turbulence intensity
    for(int i = 0; i < num_turbs - 1; i++){
        // compute the parameters
        ct = 1 / (0.00000547581845 * pow(wind_vels[threadId][i], 5.00641402) + 1.132584887);
        kk = 0.11 * pow(ct, 1.07) * pow(turb_ints[threadId][i], 0.20);
        eps = 0.23 * pow(ct, -0.25) * pow(turb_ints[threadId][i], 0.17);
        a = 0.93 * pow(ct, -0.75) * pow(turb_ints[threadId][i], 0.17);
        b = 0.42 * pow(ct, 0.6) * pow(turb_ints[threadId][i], 0.2);
        c = 0.15 * pow(ct, -0.25) * pow(turb_ints[threadId][i], -0.7);
        d = 2.3 * pow(ct, -1.2);
        e = pow(turb_ints[threadId][i], 0.1);
        f = 0.7 * pow(ct, -3.2) * pow(turb_ints[threadId][i], -0.45);
        // k1 and k2 relate to y, hence they are not handled here
        
        // solve wind velocity and turbulence intensity for each wind turbine
        for(int j = i + 1; j < num_turbs; j++){
            // here is x/d as can be found in function {rotate_and_order}
            x_d = inputs[j] - inputs[i];
            r = inputs[j + num_turbs] - inputs[i + num_turbs];

            // compute wind velocity
            delta = (*rad_addr) * (kk * x_d + eps);
            // affected wind velocity equals to the original wind velocity minus delta U
            wind_vels[threadId][j] = pow(
                pow(wind_vels[threadId][j], 2) - pow(wind_vels[threadId][i] / pow(a + b * x_d + c * pow(1 + x_d, -2), 2) * exp(-pow(r / delta, 2) / 2), 2), 0.5
            );

            // compute turbulence intensity
            k1 = (r / *rad_addr > 0.5) ? 1.0 : pow(cos(pi / 2 * (r / *rad_addr - 0.5)), 2);
            k2 = (r / *rad_addr > 0.5) ? 0.0 : pow(cos(pi / 2 * (r / *rad_addr + 0.5)), 2);
            turb_ints[threadId][j] = pow(
                pow(turb_ints[threadId][j], 2) + pow((k1 * exp(-pow((r - *rad_addr / 2) / delta, 2) / 2) + k2 * exp(-pow((r + *rad_addr / 2) / delta, 2) / 2)) / (d + e * x_d + f * pow(1 + x_d, -2)), 2), 0.5
            );
        }
    }

    // use kk as a temp variable here
    kk = 0.15 * (*cp_addr) * pi * (*rad_addr) * (*rad_addr) * (*prob_addr);  //0.5 Cp Rhop Pi R2
    // calculate energy output of wind turbines
    for(int i = 0; i < num_turbs; i++){
        // consider start wind and cut-off wind
        wind_vels[threadId][i] = wind_vels[threadId][i] < (*start_vel_addr) ? 0 : wind_vels[threadId][i];
        wind_vels[threadId][i] = wind_vels[threadId][i] > (*cut_vel_addr) ? (*cut_vel_addr) : wind_vels[threadId][i];
        // accumulate energy output
        energys[threadId] += kk * pow(wind_vels[threadId][i], 3);
    }
}

/* -------------------------------------------------------- */
// Name: plot_field_turb
// TODO: plot wind velocity and turbulence intensity field
/* -------------------------------------------------------- */
__global__ void plot_field_turb(float* inputs, float* x_array, float* y_array, float* winds, float* turbs,
    int* num_points_addr, float* plot_wind, float* plot_turb, float* rad_addr){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId >= *num_points_addr){
        // too much threads
        return;
    }

    // if(threadId == 0){
    //     // printf("Wind velocity, turbulence intensity:\n");
    //     // for(int i = 0; i < num_turbs; i++){
    //     //     printf("%f, %f\n", wind_vels[0][i], turb_ints[0][i]);
    //     // }
    //     printf("wind: %f, ia: %f, ct: %f, d: %f\n", *plot_wind, *plot_turb, *ct_addr, *rad_addr);
    // }

    // declare them in registers to accelerate the codes
    float x = x_array[threadId], y = y_array[threadId], wind = pow(*plot_wind, 2), turb = pow(*plot_turb, 2);

    // analytical model parameters
    float ct, kk, eps, a, b, c, d, e, f, k1, k2, delta, x_d, r;
    for(int i = 0; i < num_turbs; i++){
        // loop each turbine to accumulate their effects
        if(x > inputs[i]){
            // has wake effects
            // compute the parameters
            ct = 1 / (0.00000547581845 * pow(wind_vels[0][i], 5.00641402) + 1.132584887);
            kk = 0.11 * pow(ct, 1.07) * pow(turb_ints[0][i], 0.2);
            eps = 0.23 * pow(ct, -0.25) * pow(turb_ints[0][i], 0.17);
            a = 0.93 * pow(ct, -0.75) * pow(turb_ints[0][i], 0.17);
            b = 0.42 * pow(ct, 0.6) * pow(turb_ints[0][i], 0.2);
            c = 0.15 * pow(ct, -0.25) * pow(turb_ints[0][i], -0.7);
            d = 2.3 * pow(ct, -1.2);
            e = pow(turb_ints[0][i], 0.1);
            f = 0.7 * pow(ct, -3.2) * pow(turb_ints[0][i], -0.45);
            x_d = (x - inputs[i]) / (*rad_addr);
            r = y - inputs[i + num_turbs];
            delta = (*rad_addr) * (kk * x_d + eps);
            // update wind velocity
            wind -= pow(wind_vels[0][i] / pow(a + b * x_d + c * pow(1 + x_d, -2), 2) * exp(-0.5 * pow(r / delta, 2)), 2);
            // update turbulence intensity
            k1 = (r / *rad_addr > 0.5) ? 1.0 : pow(cos(pi / 2 * (r / *rad_addr - 0.5)), 2);
            k2 = (r / *rad_addr > 0.5) ? 0.0 : pow(cos(pi / 2 * (r / *rad_addr + 0.5)), 2);
            turb += pow((k1 * exp(-pow((r - *rad_addr / 2) / delta, 2) / 2) + k2 * exp(-pow((r + *rad_addr / 2) / delta, 2) / 2)) / (d + e * x_d + f * pow(1 + x_d, -2)), 2);
        }
    }

    // transfer out
    winds[threadId] = pow(wind, 0.5); turbs[threadId] = pow(turb, 0.5);
}
}