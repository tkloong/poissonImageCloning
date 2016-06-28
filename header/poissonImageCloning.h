#pragma once
#ifndef _POISSONIMAGECLONING_H
#define _POISSONIMAGECLONING_H

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
);

#endif

