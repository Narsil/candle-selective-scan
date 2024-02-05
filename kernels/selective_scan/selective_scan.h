/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMScanParamsBase {
    using index_t = uint32_t;

    int batch, seqlen, n_chunks;
    index_t a_batch_stride;
    index_t b_batch_stride;
    index_t out_batch_stride;

    // Common data pointers.
    void *__restrict__ a_ptr;
    void *__restrict__ b_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;
    bool is_variable_B;
    bool is_variable_C;

    bool delta_softplus;

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_z_batch_stride;
    index_t out_z_d_stride;

    // Common data pointers.
    void *__restrict__ A_ptr;
    void *__restrict__ B_ptr;
    void *__restrict__ C_ptr;
    void *__restrict__ D_ptr;
    void *__restrict__ u_ptr;
    void *__restrict__ delta_ptr;
    void *__restrict__ delta_bias_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
    void *__restrict__ z_ptr;
    void *__restrict__ out_z_ptr;
};

void print(SSMParamsBase &params){
    std::cout<<"SSMParamsBase{"<<std::endl;
    std::cout<<"  batch="<<params.batch<<std::endl;
    std::cout<<"  dim="<<params.dim<<std::endl;
    std::cout<<"  seqlen="<<params.seqlen<<std::endl;
    std::cout<<"  dstate="<<params.dstate<<std::endl;
    std::cout<<"  n_groups="<<params.n_groups<<std::endl;
    std::cout<<"  n_chunks="<<params.n_chunks<<std::endl;
    std::cout<<"  dim_ngroups_ratio="<<params.dim_ngroups_ratio<<std::endl;
    std::cout<<"  is_variable_B="<<params.is_variable_B<<std::endl;
    std::cout<<"  is_variable_C="<<params.is_variable_C<<std::endl;
    std::cout<<"  delta_softplus="<<params.delta_softplus<<std::endl;
    std::cout<<std::endl;
    std::cout<<"  A_d_stride="<<params.A_d_stride<<std::endl;
    std::cout<<"  A_dstate_stride="<<params.A_dstate_stride<<std::endl;
    std::cout<<std::endl;
    std::cout<<"  B_batch_stride="<<params.B_batch_stride<<std::endl;
    std::cout<<"  B_d_stride="<<params.B_d_stride<<std::endl;
    std::cout<<"  B_dstate_stride="<<params.B_dstate_stride<<std::endl;
    std::cout<<"  B_group_stride="<<params.B_group_stride<<std::endl;
    std::cout<<std::endl;
    std::cout<<"  C_batch_stride="<<params.C_batch_stride<<std::endl;
    std::cout<<"  C_d_stride="<<params.C_d_stride<<std::endl;
    std::cout<<"  C_dstate_stride="<<params.C_dstate_stride<<std::endl;
    std::cout<<"  C_group_stride="<<params.C_group_stride<<std::endl;
    std::cout<<std::endl;
    std::cout<<"  u_batch_stride="<<params.u_batch_stride<<std::endl;
    std::cout<<"  u_d_stride="<<params.u_d_stride<<std::endl;
    std::cout<<std::endl;
    std::cout<<"  delta_batch_stride="<<params.delta_batch_stride<<std::endl;
    std::cout<<"  delta_d_stride="<<params.delta_d_stride<<std::endl;
    std::cout<<std::endl;
    std::cout<<"  z_batch_stride="<<params.z_batch_stride<<std::endl;
    std::cout<<"  z_d_stride="<<params.z_d_stride<<std::endl;
    std::cout<<std::endl;
    std::cout<<"  out_batch_stride="<<params.out_batch_stride<<std::endl;
    std::cout<<"  out_d_stride="<<params.out_d_stride<<std::endl;
    std::cout<<std::endl;
    std::cout<<"  out_z_batch_stride="<<params.out_z_batch_stride<<std::endl;
    std::cout<<"  out_z_d_stride="<<params.out_z_d_stride<<std::endl;
    std::cout<<"}"<<std::endl;
}

struct SSMParamsBwd: public SSMParamsBase {
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dA_d_stride;
    index_t dA_dstate_stride;
    index_t dB_batch_stride;
    index_t dB_group_stride;
    index_t dB_d_stride;
    index_t dB_dstate_stride;
    index_t dC_batch_stride;
    index_t dC_group_stride;
    index_t dC_d_stride;
    index_t dC_dstate_stride;
    index_t du_batch_stride;
    index_t du_d_stride;
    index_t dz_batch_stride;
    index_t dz_d_stride;
    index_t ddelta_batch_stride;
    index_t ddelta_d_stride;

    // Common data pointers.
    void *__restrict__ dout_ptr;
    void *__restrict__ dA_ptr;
    void *__restrict__ dB_ptr;
    void *__restrict__ dC_ptr;
    void *__restrict__ dD_ptr;
    void *__restrict__ du_ptr;
    void *__restrict__ dz_ptr;
    void *__restrict__ ddelta_ptr;
    void *__restrict__ ddelta_bias_ptr;
};
