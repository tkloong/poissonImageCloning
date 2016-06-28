#include "../header/common.h"
#include "../header/poissonImageCloning.h"
#include <cstdio>
#include <time.h>

__global__ void PoissonImageEdit(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox,
	int stride
)
{
#define BOUND(x, h, w) \
	(((x) >= 0) && ((x) < (h)*(w)*3))
#define BOUND_MASK(x, h, w) \
	(((x) >= 0) && ((x) < (h)*(w))) 

	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	const int pt = (y*stride)*wt + x * stride;
	const int pb = (y*stride+oy)*wb + (x*stride+ox);

	int numberNeighbor = 0;
	int tNeighbors[4] = {-stride, stride, -stride*wt, stride*wt};
	bool tWeights[4] = {0, 0, 0, 0};

	// Out of range
	if (x*stride >= wt || y*stride >= ht) return;

	// Calculate number of neighbors
	for (int i=0; i<4; ++i) {
		if ( BOUND_MASK(pt + tNeighbors[i], ht, wt) && *(mask + pt + tNeighbors[i]) > 127.0f) {
			tWeights[i] = 1;
			numberNeighbor++;
		}
	}
	
	// All neighbors are black pixel out
	if (numberNeighbor == 0) return;

	float t0, t1, t2, t3, t4;
	float b1, b2, b3, b4;
	float o1, o2, o3, o4;
	double prev[3];
	double error[3];
	double totalError = 999.0;

	// Boundary
	for (int n=0; n<200 && totalError > 0.005; ++n) {
		memset(prev, 0, 3*sizeof(double));
		totalError = 0;
		if (numberNeighbor < 4 && *(mask + pt) > 127.0f ) {
			for (int i=0; i<3; ++i) {
				b1 = BOUND(pb-stride, hb, wb) ? *(background + (pb - stride)*3 + i) : 0;
				b2 = BOUND(pb+stride, hb, wb) ? *(background + (pb + stride)*3 + i) : 0;
				b3 = BOUND(pb-stride*wb, hb, wb) ? *(background + (pb - stride*wb)*3 + i) : 0;
				b4 = BOUND(pb+stride*wb, hb, wb) ? *(background + (pb + stride*wb)*3 + i) : 0;

				*(output + (pb)*3 + i) = ((!tWeights[0])*b1 + (!tWeights[1])*b2 + (!tWeights[2])*b3 
						+ (!tWeights[3])*b4)/(4-numberNeighbor);
			}
			for (int i=0; i<3; ++i) {
				totalError += error[i] * error[i];
			}
		}
		// Interior
		else if (*(mask + pt) > 127.0f){
			for (int i=0; i<3; ++i) {
				t0 = BOUND(pt, ht, wt) ? *(target + (pt)*3 + i) : 0;
				t1 = BOUND(pt-stride, ht, wt) ? *(target + (pt - stride)*3 + i) : 0;
				t2 = BOUND(pt+stride, ht, wt) ? *(target + (pt + stride)*3 + i) : 0;
				t3 = BOUND(pt-stride*wt, ht, wt) ? *(target + (pt - stride*wt)*3 + i) : 0;
				t4 = BOUND(pt+stride*wt, ht, wt) ? *(target + (pt + stride*wt)*3 + i) : 0;

				b1 = BOUND(pb-stride, hb, wb) ? *(background + (pb - stride)*3 + i) : 0;
				b2 = BOUND(pb+stride, hb, wb) ? *(background + (pb + stride)*3 + i) : 0;
				b3 = BOUND(pb-stride*wb, hb, wb) ? *(background + (pb - stride*wb)*3 + i) : 0;
				b4 = BOUND(pb+stride*wb, hb, wb) ? *(background + (pb + stride*wb)*3 + i) : 0;

				o1 = BOUND(pb-stride, hb, wb) ? *(output + (pb - stride)*3 + i) : 0;
				o2 = BOUND(pb+stride, hb, wb) ? *(output + (pb + stride)*3 + i) : 0;
				o3 = BOUND(pb-stride*wb, hb, wb) ? *(output + (pb - stride*wb)*3 + i) : 0;
				o4 = BOUND(pb+stride*wb, hb, wb) ? *(output + (pb + stride*wb)*3 + i) : 0;

				error[i] = prev[i] - *(output + pb*3 + i);
				*(output + (pb)*3 + i) = ((4*t0 - (t1 + t2 + t3 + t4)) + (o1 + o2 + o3 + o4))/4;
				prev[i] = *(output + pb*3 + i);
			}
			for (int i=0; i<3; ++i) {
				totalError += error[i] * error[i];
			}
		}
	}
}

__global__ void scaleUp(
		const float *mask,
		float *output, 
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox,
		int stride
		)
{
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	const int pt = y*wt + x;
	const int pb = (y+oy)*wb + (x+ox);

	if (x >= wt || y >= ht || *(mask + pt) < 127.0f) return;

	int pr = ((y/stride)*stride+oy)*wb + ((x/stride)*stride+ox);
	for (int i=0; i<3; i++)
		*(output + pb*3 + i) = *(output + pr*3 + i);
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);

	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
			background, target, mask, output,
			wb, hb, wt, ht, oy, ox
			);

	clock_t t = clock();

	for (int scale=16; scale>1; scale>>=1) {
		PoissonImageEdit<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
				background, target, mask, output,
				wb, hb, wt, ht, oy, ox, scale
				);
		scaleUp<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
				mask, output, wb, hb, wt, ht, oy, ox, scale
				);
	}

	PoissonImageEdit<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
			background, target, mask, output,
			wb, hb, wt, ht, oy, ox, 1
			);

	printf("Time spent: %lf\n", (double)(clock() - t)/CLOCKS_PER_SEC);
}
