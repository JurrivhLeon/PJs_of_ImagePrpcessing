import numpy as np


class FFD:
    def __init__(self, lx=64, ly=64, ncpx=9, ncpy=9):
        # Number of pixels in each grid. (length of intervals)
        self.lx = lx
        self.ly = ly
        # Number of control points in each direction.
        self.ncpx = ncpx
        self.ncpy = ncpy
        # Original coordinates of control points.
        x = np.linspace(0, lx * (ncpx - 1), ncpx)
        y = np.linspace(0, ly * (ncpy - 1), ncpy)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.cp_org = np.stack((xx, yy), axis=-1)
        # Offset of control points.
        self.cp_offset = np.zeros((ncpx, ncpy, 2))

    @staticmethod
    def Bspline(u: np.float, idx: int):
        # B-spline function.
        if idx == 0:
            return (1 - u) ** 3 / 6
        elif idx == 1:
            return (3 * u ** 3 - 6 * u ** 2 + 4) / 6
        elif idx == 2:
            return (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6
        else:
            return u ** 3 / 6

    def update_offset(self, mat):
        # Read in the offset of control points from a text file.
        print('Read in offset data.')
        with open(mat, 'r') as fp:
            txt = fp.read()
            rows = txt.split('\n')
            i = 0
            for row in rows:
                j = 0
                cols = row.split(',')
                for col in cols:
                    self.cp_offset[i, j, :] = np.array(col.split()).astype(float)
                    j += 1
                i += 1

    def FFDtransform(self, coord: np.float):
        """
        2D FFD transform.
        Input: Coordinate matrix of shape 2 * N, each column corresponds to a pixel point.
        Output: Transformed coordinate matrix.
        """
        x, y = coord[0, :], coord[1, :]
        flr_x, flr_y = np.floor([x / self.lx, y / self.ly]).astype(int)

        # Lattice coordinates.
        u, v = x / self.lx - flr_x, y / self.ly - flr_y

        q_local = np.zeros_like(coord)
        # Get the proximal control point.
        get_cp = lambda a, b: self.cp_offset[a.clip(0, self.ncpx - 1), b.clip(0, self.ncpy - 1)].T
        # Local offset.
        for i in range(-1, 3):
            for j in range(-1, 3):
                bunch = self.Bspline(u, idx=i+1) * self.Bspline(v, idx=j+1)
                bunch = np.array([bunch] * 2)
                q_local += bunch * get_cp(flr_x + i, flr_y + j)
        return coord + q_local

    def compute_gradient(self, x, y, xt, yt, h=5e-3):
        """
        Use finite-difference approach to estimate the gradient:
        df/dx = [f(x + h) - f(x - h)] / (2 * h)
        Here the objective function f is the squared distance between
        transformed pixel point and objective pixel point.
        Input: (x, y) -> current point, (xt, yt) -> transformed objective point;
        Output: column vector (df/dx, df/dy).
        """
        f = lambda x, y, x0, y0: ((self.FFDtransform(np.array([[x, y]]).T) - np.array([[x0, y0]]).T) ** 2).sum()
        dfdx = (f(x + h, y, xt, yt) - f(x - h, y, xt, yt)) / (2 * h)
        dfdy = (f(x, y + h, xt, yt) - f(x, y - h, xt, yt)) / (2 * h)
        return np.array([[dfdx, dfdy]]).T

    def compute_hessian(self, x, y, xt, yt, h=5e-3):
        """
        Use finite-difference approach to estimate the hessian matrix:
        d2f/d2x = [f'(x + h) - f'(x - h)] / (2 * h)
        Here the objective function f is the squared distance between
        transformed pixel point and objective pixel point.
        Input: (x, y) -> current point, (xt, yt) -> transformed objective point;
        Output: Hessian matrix (d2f/dx2, d2f/dxdy; d2f/dxdy, d2f/dy2).
        """
        f = lambda x, y, x0, y0: ((self.FFDtransform(np.array([[x, y]]).T) - np.array([[x0, y0]]).T) ** 2).sum()
        d2fdx2 = (f(x + 2 * h, y, xt, yt) + f(x - 2 * h, y, xt, yt) - 2 * f(x, y, xt, yt)) / (4 * h * h)
        d2fdy2 = (f(x, y + 2 * h, xt, yt) + f(x, y - 2 * h, xt, yt) - 2 * f(x, y, xt, yt)) / (4 * h * h)
        d2fdxdy = (f(x + h, y + h, xt, yt) - f(x + h, y - h, xt, yt) - f(x - h, y + h, xt, yt) + f(x - h, y - h, xt, yt)) / (4 * h * h)
        return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]])

    def optim_gd(self, origin, goal, lr=2, epsilon=1, alpha=1e-04):
        """
        Use gradient descent method for optimization.
        Compute the point, which is mapped onto the goal point after FFD transformation.
        Input:  origin -> start point, goal -> a,
                lr -> learning rate, epsilon: acceptable error;
        Output: optimal solution (x*, y*), error, number of iterations.
        """
        # Initialization.
        origin, goal = origin.reshape((2, 1)), goal.reshape((2, 1))
        error = ((self.FFDtransform(origin) - goal) ** 2).sum()
        p = origin
        ite = 0
        # Iterations.
        while error > epsilon and ite < 1e3:
            grad = self.compute_gradient(p[0, 0], p[1, 0], goal[0, 0], goal[1, 0])
            s = 1
            p_new, error_new = np.zeros_like(p), error
            # Use Armijo line search.
            while error_new - error + alpha * lr * s * (grad ** 2).sum() > 0:
                p_new = p - lr * grad * s
                error_new = ((self.FFDtransform(p_new) - goal) ** 2).sum()
                s *= 0.5
            if np.linalg.norm(p - p_new) < 1e-2:
                break
            p, error = p_new, error_new
            ite += 1
        return {'sol': p, 'error': error, 'ite': ite}

    def optim_newton(self, origin, goal, lr=1, epsilon=1, alpha=1e-04):
        """
        Use newton method for optimization.
        Compute the point, which is mapped onto the goal point after FFD transformation.
        Input:  origin -> start point, goal -> a,
                lr -> learning rate, epsilon: acceptable error;
        Output: optimal solution (x*, y*), error, number of iterations.
        """

        # Initialization.
        origin, goal = origin.reshape((2, 1)), goal.reshape((2, 1))
        error = ((self.FFDtransform(origin) - goal) ** 2).sum()
        p = origin
        ite = 0
        # Iterations.
        while error > epsilon and ite < 1e3:
            grad = self.compute_gradient(p[0, 0], p[1, 0], goal[0, 0], goal[1, 0])
            hess = self.compute_hessian(p[0, 0], p[1, 0], goal[0, 0], goal[1, 0])
            decrement = np.linalg.solve(hess, grad)
            s = 1
            p_new, error_new = np.zeros_like(p), error
            # Use Armijo line search.
            while error_new - error + alpha * s * lr * (grad * decrement).sum() > 0:
                p_new = p - lr * decrement * s
                error_new = ((self.FFDtransform(p_new) - goal) ** 2).sum()
                s *= 0.5
            if np.linalg.norm(p - p_new) < 1e-2:
                break
            p, error = p_new, error_new
            ite += 1
        return {'sol': p, 'error': error, 'ite': ite}

    def FFDinvtransform(self, coord: np.float):
        """
        Based on given coordinate matrix, solve the original coordinate matrix before FFD.
        Input: transformed coordinate matrix of shape 2 * N;
        Output: original matrix of shape 2 * N, mean square error.
        """
        # Initialization: compute an approximate solution.
        origin = 2 * coord - self.FFDtransform(coord)

        # Compute the inverse.
        inv = np.zeros_like(origin)
        mse = 0
        sum_iter = 0
        stone = coord.shape[1] // 100
        for i in range(coord.shape[1]):
            opt = self.optim_newton(origin[:, i], coord[:, i])
            inv[:, i] = opt['sol'].flatten()
            mse += opt['error']
            sum_iter += opt['ite']
            if i % stone == 0:
                print('Computing inverse,\titerations: {0:7d},\t{1}% completed;'.format(sum_iter, i / stone))
        mse = np.sqrt(mse / coord.shape[1])
        return {'inv': inv, 'mse': mse}
