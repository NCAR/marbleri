#
# Calculate z-component of vorticity
#  (only does non-boundary points since point of routine is for center finding
def vortz(u, v, x, y):
	dvdx = 0 * v
	dudy = 0 * u
	[nx, ny] = u.shape
	dvdx[1:nx-1, 1:ny-1] = ((v[1:nx-1,2:ny] - v[1:nx-1, 0:ny-2]) /
		(x[1:nx-1, 2:ny] - x[1:nx-1, 0:ny-2]))
	dudy[1:nx-1, 1:ny-1] = ((u[0:nx-2, 1:ny-1] - u[2:nx, 1:ny-1]) / 
		(y[0:nx-2, 1:ny-1] - y[2:nx, 1:ny-1]))
	vort = dvdx - dudy
	return vort
