__kernel void
sampleKernel(__global const double *a,
             __global const double *b,
             __global double *c)
{
    int gid = get_global_id(0);
    c[gid] = a[gid] * b[gid];
}