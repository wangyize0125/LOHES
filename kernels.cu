#include <stdio.h>

extern "C"{
// global variables
// ratio between radian and angle
__device__ const float pi = 3.1415926535;
__device__ const float pi_ratio = pi / 180;
__device__ const float epsilon = 1e-10;
// number of individuals
__device__ const int num_inds = ${num_inds};
// number of devices
__device__ const int num_turbs = ${num_turbs};
// number of converters
__device__ const int num_converters = ${num_converters};
// wind velocities
__device__ float wind_vels[num_inds][num_turbs];
// turbulence intensities
__device__ float turb_ints[num_inds][num_turbs];
// wave heights
__device__ float wave_heights[num_inds][num_converters];
// parameters of the wake model
__device__ float paras_all[12 * num_inds];

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
    return is_ok? 100000 / cv_true: 100000 / cv_false;
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

/* -------------------------------------------------------- */
// Name: sort_converters
// TODO: sort the wave energy converters in ascending order
/* -------------------------------------------------------- */
__global__ void sort_converters(float* inputs_all){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(threadId >= num_inds){
        // too much threads
        return;
    }
    
    // change inputs for the current kernel function
    float *inputs = inputs_all + threadId * num_converters * 2;
    // two temp variables
    float angle, dist;
    // sort the wind turbines in ascending order of x
    for(int i = 0; i < num_converters - 1; i++){
        for(int j = 0; j < num_converters - i - 1; j++){
            if(inputs[j] > inputs[j + 1]){
                // to reduce the memory cost of this function
                // use angle and dist as temp variables here
                angle = inputs[j]; dist = inputs[j + num_converters]; 
                inputs[j] = inputs[j + 1]; inputs[j + num_converters] = inputs[j + 1 + num_converters];
                inputs[j + 1] = angle; inputs[j + 1 + num_converters] = dist;
            }
        }
    }
}

/* -------------------------------------------------------- */
// Name: cal_cv_pre_turb
// TODO: calculate constraint values
/* -------------------------------------------------------- */
__device__ inline float cal_cv_pre_turb(float* inputs, float sx, float sy, int num_devs, float* pre_layouts, int num_pre_layouts){
    // varibale indicating whether it is ok for this layout
    bool is_ok = true;

    float x, y, cv_true = 0, cv_false = 0;
    // loop each wave energy converter to check
    for(int i = 0; i < num_devs; i++){
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

        // j means the jth wind turbine
        for(int j = 0; j < num_pre_layouts; j++){
            // calculate spacing between them
            x = abs(pre_layouts[j] - inputs[i]);
            y = abs(pre_layouts[j + num_pre_layouts] - inputs[i + num_devs]);

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

    // use 100000/* to ensure they are compact
    return is_ok? 100000 / cv_true: 100000 / cv_false;
}

/* -------------------------------------------------------- */
// Name: cal_cv_converter
// TODO: calculate constraint values of wave energy converters
/* -------------------------------------------------------- */
__global__ void cal_cv_converter(float* inputs, float* cvs, float* sx, float* sy, float* pre_layouts, int* num_pre_layouts){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId >= num_inds){
        // too much threads
        return;
    }

    cvs[threadId] = cal_cv_pre_turb(inputs + threadId * num_converters * 2, *sx, *sy, num_converters, pre_layouts, *num_pre_layouts);
}

/* -------------------------------------------------------- */
// Name: pre_energy_converter
// TODO: calculate energy outputs of the wave energy converters 
/* -------------------------------------------------------- */
__global__ void pre_energy_converter(float* inputs, float* energys, float* periods, int* num_t,
    float* heights, int* num_h, float* probs, float* wake_model_first, float* energy_model, 
    float* pre_layouts, int* num_pre_layouts, float* heights_wind_turb, float* temp){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId >= num_inds){
        // too much threads
        return;
    }

    // set zero
    energys[threadId * 2] = energys[threadId * 2 + 1] = 0.0;

    // re-locate inputs for the current thread
    inputs += threadId * num_converters * 2;
    heights_wind_turb += (*num_pre_layouts) * threadId;
    for(int i = 0; i < (*num_pre_layouts); i++){
        heights_wind_turb[i] = 0;
    }
    temp += (*num_pre_layouts) * threadId;

    float deficit, x, y, period, height, prob;
    // for convinience in accessing those data, re-locate them for the current thread
    float *paras = paras_all + threadId * 12, *wake_model;
    // iterate all the wave height and period combinations
    for(int t = 0; t < (*num_t); t++){
        for(int h = 0; h < (*num_h); h++){
            // no wave
            if(probs[h * (*num_t) + t] <= 1e-10) continue;
            else{
                // has wave
                // pick out period, height, and probability first
                period = periods[t]; height = heights[(*num_h) - 1 - h]; prob = probs[h * (*num_t) + t];
                
                // correct the period that is out of range
                if(period < 5.5) period = 5.5;
                else if(period > 11.5) period = 11.5;
                else period = period;

                // initialize wave height array
                for(int i = 0; i < num_converters; i++){
                    wave_heights[threadId][i] = height;
                }
                for(int i = 0; i < (*num_pre_layouts); i++){
                    temp[i] = height;
                }
            
                // iterate all the wave energy converters
                for(int i = 0; i < num_converters; i++){
                    // solve the wake model first
                    for(int j = 0; j < 12; j++){
                        // locate wake model for this parameter
                        wake_model = wake_model_first + 6 * j;
                        paras[j] = wake_model[0] * period * period + wake_model[1] * wave_heights[threadId][i] * wave_heights[threadId][i] + \
                                   wake_model[2] * period * wave_heights[threadId][i] + wake_model[3] * period + wake_model[4] * wave_heights[threadId][i] + wake_model[5];
                    }

                    // solve wave height for each wave energy converter
                    for(int j = i + 1; j < num_converters; j++){
                        // calculate distance between two converters
                        x = inputs[j] - inputs[i];
                        y = inputs[j + num_converters] - inputs[i + num_converters];

                        // calculate the wave deficit percentage
                        deficit = pow(abs(paras[0]) + x, -abs(paras[1])) * (exp(-pow((y - paras[2] - paras[3] * x) / (paras[4] + paras[5] * x + epsilon), 2)) + \
                                  exp(-pow((y + paras[2] + paras[3] * x) / (paras[4] + paras[5] * x + epsilon), 2)));
                        // accumulate the wave surplus percentage
                        deficit -= pow(abs(paras[6]) + x, -abs(paras[7])) * (exp(-pow((y - paras[8] - paras[9] * x) / (paras[10] + paras[11] * x + epsilon), 2)) + \
                                   exp(-pow((y + paras[8] + paras[9] * x) / (paras[10] + paras[11] * x + epsilon), 2)));

                        // calculate wave height of the ith converter in the threadIdth individual
                        wave_heights[threadId][j] = pow(abs(pow(wave_heights[threadId][j], 2) - (deficit >= 0 ? 1 :-1) * pow(deficit * wave_heights[threadId][i], 2)), 0.5);
                    }

                    // solve wave height for each wind turbine
                    for(int j = 0; j < (*num_pre_layouts); j++){
                        // calculate distance between wave energy converter and wind turbine
                        x = pre_layouts[j] - inputs[i];
                        if(x <= 0) continue;
                        y = pre_layouts[j + (*num_pre_layouts)] - inputs[i + num_converters];

                        // calculate the wave deficit percentage
                        deficit = pow(abs(paras[0]) + x, -abs(paras[1])) * (exp(-pow((y - paras[2] - paras[3] * x) / (paras[4] + paras[5] * x + epsilon), 2)) + \
                                  exp(-pow((y + paras[2] + paras[3] * x) / (paras[4] + paras[5] * x + epsilon), 2)));
                        // accumulate the wave surplus percentage
                        deficit -= pow(abs(paras[6]) + x, -abs(paras[7])) * (exp(-pow((y - paras[8] - paras[9] * x) / (paras[10] + paras[11] * x + epsilon), 2)) + \
                                   exp(-pow((y + paras[8] + paras[9] * x) / (paras[10] + paras[11] * x + epsilon), 2)));

                        temp[j] = pow(abs(pow(temp[j], 2) - (deficit >= 0 ? 1 :-1) * pow(deficit * wave_heights[threadId][i], 2)), 0.5);
                    }
                }

                if(periods[t] < 3.5){
                    // set the initial wave heights to be incoming wave heights
                    // do not consider wake effects herein
                    for(int i = 0; i < num_converters; i++){
                        wave_heights[threadId][i] = height;
                    }
                }
                
                // calculate energy output
                for(int i = 0; i < num_converters; i++){
                    energys[threadId * 2] += (energy_model[0] * period * period + energy_model[1] * wave_heights[threadId][i] * wave_heights[threadId][i] + \
                                             energy_model[2] * period * wave_heights[threadId][i] + energy_model[3] * period + energy_model[4] * wave_heights[threadId][i] + energy_model[5]) * prob;
                }
                // calculate wave height ahead wind turbines
                for(int i = 0; i < (*num_pre_layouts); i++){
                    heights_wind_turb[i] += temp[i] * prob;
                }
            }
        }
    }

    // before, I have computed the wave heights ahead each wind turbine. However, I am wondering how to integrate them
    // into one variable. I have two ways with one of them finding out the maximum value and the other one summing up them
    // all. When I use this two different ways, I found that they have different application situation. For the first one,
    // if the number of wave energy converters is larger than that of wind turbines, it is ok for me to find out the maximum 
    // value. However, if the number of wave energy converters is smaller than that of wind turbines, the maximum wave height
    // will always be the incoming wave height. Hence, in this case, summing them up is effective
    if(num_converters >= (*num_pre_layouts)){
        for(int i = 0; i < (*num_pre_layouts); i++){
            energys[threadId * 2 + 1] = energys[threadId * 2 + 1] > heights_wind_turb[i] ? energys[threadId * 2 + 1] : heights_wind_turb[i];
        }
    }
    else{
        for(int i = 0; i < (*num_pre_layouts); i++){
            energys[threadId * 2 + 1] += heights_wind_turb[i];
        }
    }
}

/* -------------------------------------------------------- */
// Name: cal_cv_turbine_converter
// TODO: calculate cv of wind turbine and wave energy converter 
/* -------------------------------------------------------- */
__global__ void cal_cv_turbine_converter(float* turbines, float* converters, float* cvs, float* sx_turb, float* sy_turb, float* sx_conv, float* sy_conv){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId >= num_inds){
        // too much threads
        return;
    }

    // re-locate 
    turbines += threadId * num_turbs * 2; converters += threadId * num_converters * 2;

    // varibale indicating whether it is ok for this layout
    bool is_ok = true;

    float x, y, cv_true = 0, cv_false = 0;
    // loop each wave energy converter to check
    for(int i = 0; i < num_converters; i++){
        // converter to converter
        for(int j = i + 1; j < num_converters; j++){
            // calculate spacing between them
            x = abs(converters[j] - converters[i]);
            y = abs(converters[j + num_converters] - converters[num_converters + i]);

            // check
            if(x > *sx_conv || y > *sy_conv){
                cv_true -= (x + y);
            }
            else{
                cv_false += (x + y);
                is_ok = false;
            }
        }

        // converter to turbine
        for(int j = 0; j < num_turbs; j++){
            // calculate spacing between them
            x = abs(turbines[j] - converters[i]);
            y = abs(turbines[j + num_turbs] - converters[i + num_converters]);

            if(x > *sx_conv || y > *sy_conv){
                cv_true -= (x + y);
            }
            else{
                cv_false += (x + y);
                is_ok = false;
            }
        }
    }

    // loop each wind turbine to check
    for(int i = 0; i < num_turbs - 1; i++){
        for(int j = i + 1; j < num_turbs; j++){
            // calculate spacing between them
            x = abs(turbines[j] - turbines[i]);
            y = abs(turbines[j + num_turbs] - turbines[i + num_turbs]);

            if(x > *sx_turb || y > *sy_turb){
                cv_true -= (x + y);
            }
            else{
                cv_false += (x + y);
                is_ok = false;
            }
        }
    }

    // use 100000/* to ensure they are compact
    cvs[threadId] = is_ok? 100000 / cv_true: 100000 / cv_false;
}

/* -------------------------------------------------------- */
// Name: pre_energy_turbine_converter
// TODO: predict energy of wind turbine and wave energy converter 
/* -------------------------------------------------------- */
__global__ void pre_energy_turbine_converter(float* converters, float* energies, float* periods, int* num_t, float* heights, int* num_h,
    float* probs, float* wake_model_first, float* energy_model, float* heights_wind_turb, float* temp, float* turbines, float* vels,
    int* num_vel, float* direcs, int* num_direc, float* wind_dist, float* turb_int_addr, float* rot_turbine, float* radius, float* cp_addr, 
    float* start_vel_addr, float* cut_vel_addr){
    // calculate the index
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId >= num_inds){
        // too much threads
        return;
    }

    // set zero
    energies[threadId * 2] = energies[threadId * 2 + 1] = 0.0;

    // re-locate
    converters += threadId * num_converters * 2; turbines += threadId * num_turbs * 2;
    heights_wind_turb += num_turbs * threadId; temp += num_turbs * threadId;
    for(int i = 0; i < num_turbs; i++){
        heights_wind_turb[i] = 0.0;
    } 
    rot_turbine += threadId * num_turbs * 2;

    float deficit, x, y, period, height, prob;
    // for convinience in accessing those data, re-locate them
    float *paras = paras_all + threadId * 12, *wake_model;
    // loop all the wave height and period combinations
    for(int t = 0; t < (*num_t); t++){
        for(int h = 0; h < (*num_h); h++){
            // no wave
            if(probs[h * (*num_t) + t] < 1e-10) continue;
            else{
                // has wave
                // pick out period, height, and probability first
                period = periods[t]; height = heights[(*num_h) - 1 - h]; prob = probs[h * (*num_t) + t];

                // correct the period that is out of range
                if(period < 5.5) period = 5.5;
                else if(period > 11.5) period = 11.5;
                else period = period;

                // initialize wave height array
                for(int i = 0; i < num_converters; i++){
                    wave_heights[threadId][i] = height;
                }
                for(int i = 0; i < num_turbs; i++){
                    temp[i] = height;
                }

                // iterate all the wave energy converters
                for(int i = 0; i < num_converters; i++){
                    // solve the wake model first
                    for(int j = 0; j < 12; j++){
                        // locate wake model for this parameter
                        wake_model = wake_model_first + 6 * j;
                        paras[j] = wake_model[0] * period * period + wake_model[1] * wave_heights[threadId][i] * wave_heights[threadId][i] + \
                                   wake_model[2] * period * wave_heights[threadId][i] + wake_model[3] * period + wake_model[4] * wave_heights[threadId][i] + wake_model[5];
                    }

                    for(int j = i + 1; j < num_converters; j++){
                        // distance
                        x = converters[j] - converters[i];
                        y = converters[j + num_converters] - converters[i + num_converters];

                        // calculate the wave deficit percentage
                        deficit = pow(abs(paras[0]) + x, -abs(paras[1])) * (exp(-pow((y - paras[2] - paras[3] * x) / (paras[4] + paras[5] * x + epsilon), 2)) + \
                        exp(-pow((y + paras[2] + paras[3] * x) / (paras[4] + paras[5] * x + epsilon), 2)));
                        // accumulate the wave surplus percentage
                        deficit -= pow(abs(paras[6]) + x, -abs(paras[7])) * (exp(-pow((y - paras[8] - paras[9] * x) / (paras[10] + paras[11] * x + epsilon), 2)) + \
                        exp(-pow((y + paras[8] + paras[9] * x) / (paras[10] + paras[11] * x + epsilon), 2)));

                        // calculate wave height of the ith converter in the threadIdth individual
                        wave_heights[threadId][j] = pow(abs(pow(wave_heights[threadId][j], 2) - (deficit >= 0 ? 1 :-1) * pow(deficit * wave_heights[threadId][i], 2)), 0.5);
                    }

                    // solve wave height
                    for(int j = 0; j < num_turbs; j++){
                        x = turbines[j] - converters[i];
                        if(x <= 0) continue;
                        y = turbines[j + num_turbs] - converters[i + num_converters];

                        // calculate the wave deficit percentage
                        deficit = pow(abs(paras[0]) + x, -abs(paras[1])) * (exp(-pow((y - paras[2] - paras[3] * x) / (paras[4] + paras[5] * x + epsilon), 2)) + \
                        exp(-pow((y + paras[2] + paras[3] * x) / (paras[4] + paras[5] * x + epsilon), 2)));
                        // accumulate the wave surplus percentage
                        deficit -= pow(abs(paras[6]) + x, -abs(paras[7])) * (exp(-pow((y - paras[8] - paras[9] * x) / (paras[10] + paras[11] * x + epsilon), 2)) + \
                        exp(-pow((y + paras[8] + paras[9] * x) / (paras[10] + paras[11] * x + epsilon), 2)));

                        temp[j] = pow(abs(pow(temp[j], 2) - (deficit >= 0 ? 1 :-1) * pow(deficit * wave_heights[threadId][i], 2)), 0.5);
                    }
                }

                if(periods[t] < 3.5){
                    // set the initial wave heights to be incoming wave heights
                    // do not consider wake effects herein
                    for(int i = 0; i < num_converters; i++){
                        wave_heights[threadId][i] = height;
                    }
                    for(int i = 0; i < num_turbs; i++){
                        temp[i] = height;
                    }
                }

                // calculate energy output
                for(int i = 0; i < num_converters; i++){
                    energies[threadId * 2] += (energy_model[0] * period * period + energy_model[1] * wave_heights[threadId][i] * wave_heights[threadId][i] + \
                                             energy_model[2] * period * wave_heights[threadId][i] + energy_model[3] * period + energy_model[4] * wave_heights[threadId][i] + energy_model[5]) * prob * 1000;
                }
                // calculate wave height ahead wind turbines
                for(int i = 0; i < num_turbs; i++){
                    heights_wind_turb[i] += temp[i] * prob;
                }
            }
        }
    }

    // before, I have computed the wave heights ahead each wind turbine. However, I am wondering how to integrate them
    // into one variable. I have two ways with one of them finding out the maximum value and the other one summing up them
    // all. When I use this two different ways, I found that they have different application situation. For the first one,
    // if the number of wave energy converters is larger than that of wind turbines, it is ok for me to find out the maximum 
    // value. However, if the number of wave energy converters is smaller than that of wind turbines, the maximum wave height
    // will always be the incoming wave height. Hence, in this case, summing them up is effective
    if(num_converters >= num_turbs){
        for(int i = 0; i < num_turbs; i++){
            energies[threadId * 2 + 1] = energies[threadId * 2 + 1] > heights_wind_turb[i] ? energies[threadId * 2 + 1] : heights_wind_turb[i];
        }
    }
    else{
        for(int i = 0; i < num_turbs; i++){
            energies[threadId * 2 + 1] += heights_wind_turb[i];
        }
    }

    // parameters of the analytical models
    float ct, kk, eps, a, b, c, d, e, f, k1, k2, delta, x_d, r;
    // loop direction
    for(int dre = 0; dre < (*num_direc); dre++){
        // rotate the wind turbines to face the incoming winds when direction changed
        // assuming that x axis along the incoming wind direction
        float angle = 0, dist = 0;    // angle between x axis
        for(int i = 0; i < num_turbs; i++){
            // calculate the angle
            angle = atan(turbines[num_turbs + i] / turbines[i]);
            // rotate the coordinate system
            angle += direcs[dre] * pi_ratio;
            // calculate the distance
            dist = pow(pow(turbines[num_turbs + i], 2) + pow(turbines[i], 2), 0.5);
            // calculate the new coordinates
            // NOTE: here x coordinate are normalized by rotor radius owing to that x/D 
            // is used in the analytical models
            rot_turbine[i] = dist * cos(angle) / (*radius); rot_turbine[num_turbs + i] = dist * sin(angle);
        }

        // sort the wind turbines in ascending order of x
        for(int i = 0; i < num_turbs - 1; i++){
            for(int j = 0; j < num_turbs - i - 1; j++){
                if(rot_turbine[j] > rot_turbine[j + 1]){
                    // to reduce the memory cost of this function
                    // use angle and dist as temp variables here
                    angle = rot_turbine[j]; dist = rot_turbine[j + num_turbs]; 
                    rot_turbine[j] = rot_turbine[j + 1]; rot_turbine[j + num_turbs] = rot_turbine[j + 1 + num_turbs];
                    rot_turbine[j + 1] = angle; rot_turbine[j + 1 + num_turbs] = dist;
                }
            }
        }

        // loop wind velocity
        for(int v = 0; v < (*num_vel); v++){
            // set zero
            for(int i = 0; i < num_turbs; i++){
                wind_vels[threadId][i] = vels[i]; turb_ints[threadId][i] = *turb_int_addr;
            }

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
                    x_d = rot_turbine[j] - rot_turbine[i];
                    r = rot_turbine[j + num_turbs] - rot_turbine[i + num_turbs];

                    // compute wind velocity
                    delta = (*radius) * (kk * x_d + eps);
                    // affected wind velocity equals to the original wind velocity minus delta U
                    wind_vels[threadId][j] = pow(
                        pow(wind_vels[threadId][j], 2) - pow(wind_vels[threadId][i] / pow(a + b * x_d + c * pow(1 + x_d, -2), 2) * exp(-pow(r / delta, 2) / 2), 2), 0.5
                    );

                    // compute turbulence intensity
                    k1 = (r / *radius > 0.5) ? 1.0 : pow(cos(pi / 2 * (r / *radius - 0.5)), 2);
                    k2 = (r / *radius > 0.5) ? 0.0 : pow(cos(pi / 2 * (r / *radius + 0.5)), 2);
                    turb_ints[threadId][j] = pow(
                        pow(turb_ints[threadId][j], 2) + pow((k1 * exp(-pow((r - *radius / 2) / delta, 2) / 2) + k2 * exp(-pow((r + *radius / 2) / delta, 2) / 2)) / (d + e * x_d + f * pow(1 + x_d, -2)), 2), 0.5
                    );
                }
            }

            // use kk as a temp variable here
            kk = 0.15 * (*cp_addr) * pi * (*radius) * (*radius) * wind_dist[dre * (*num_vel) + v];
            // calculate energy output of wind turbines
            for(int i = 0; i < num_turbs; i++){
                // consider start wind and cut-off wind
                wind_vels[threadId][i] = wind_vels[threadId][i] < (*start_vel_addr) ? 0 : wind_vels[threadId][i];
                wind_vels[threadId][i] = wind_vels[threadId][i] > (*cut_vel_addr) ? (*cut_vel_addr) : wind_vels[threadId][i];
                // accumulate energy output
                energies[threadId * 2] += kk * pow(wind_vels[threadId][i], 3);
            }
        }
    }
}
}