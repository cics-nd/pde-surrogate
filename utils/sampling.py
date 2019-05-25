"""
Sampling in spatial domain

collocation points
boundary points

For sure, lots of people will work on how to use different sampling grid
to train fully-connected networks.

"""
import numpy as np
import torch
from .lhs import lhs


class SampleSpatial2d(object):
    """Uniform grid
    (y, x)

    h - height, or y axis
    w - width, x axis

    default output [0, 1] from [0, ngrid_h - 1], [0, ngrid_w - 1]
    """
    def __init__(self, ngrid_h, ngrid_w):
        self.ngrid_h = int(ngrid_h)
        self.ngrid_w = int(ngrid_w)
        self.n_grids = self.ngrid_h * self.ngrid_w
        self.refactor = torch.FloatTensor(np.array([[ngrid_h-1, ngrid_w-1]]))
        self.coordinates = self._coordinates()
        self.coordinates_no_boundary = self._coordinates_no_boundary()

    def _coordinates(self):
        # super wired torch.meshgrid
        grid_x, grid_y = np.meshgrid(np.arange(self.ngrid_w), np.arange(self.ngrid_h))
        points = np.stack((grid_y.flatten(), grid_x.flatten()), 1)
        return torch.FloatTensor(points)

    def _coordinates_no_boundary(self):
        grid_x, grid_y = np.meshgrid(np.arange(self.ngrid_w), np.arange(self.ngrid_h))
        points = np.stack((grid_y[1:-1, 1:-1].flatten(), grid_x[1:-1, 1:-1].flatten()), 1)
        return torch.FloatTensor(points)


    def _sample2d(self, on_grid, n_samples=None, no_boundary=False):
        if n_samples is None:
            n_samples = self.n_grids
        if on_grid:
            if no_boundary:
                points = self.coordinates_no_boundary.to(torch.float32) / self.refactor
            else:
                points = self.coordinates.to(torch.float32) / self.refactor
            if n_samples < len(points):
                points = points[torch.randperm(self.n_grids)[:n_samples]]
            else:
                print('n_samples is greater than grid size, set n_samples '\
                    'equals to grid size')
        else:
            points = torch.FloatTensor(lhs(2, n_samples))
        
        return points
                

    def _sample1d(self, horizontal, on_grid, n_samples=None):
        """
        if on_grid is on, n_sampels is ignored if it is larger than ngrid.
        """
        ngrid = self.ngrid_h if horizontal else self.ngrid_w
        if n_samples is None:
            n_samples = ngrid
        if on_grid:
            points = (torch.arange(float(ngrid)) / (ngrid-1))
            if n_samples <= len(points):
                points = points[torch.randperm(ngrid)[:n_samples]]
            else:
                print('n_samples is greater than grid size, set n_samples '\
                    'equals to grid size')
        else:
            points = torch.rand(n_samples)
        return points

    def left(self, on_grid=True, n_samples=None):
        points = self._sample1d(horizontal=True, on_grid=on_grid, n_samples=n_samples)
        return torch.stack((points, torch.zeros_like(points)), 1)

    def right(self, on_grid=True, n_samples=None):
        points = self._sample1d(horizontal=True, on_grid=on_grid, n_samples=n_samples)
        return torch.stack((points, torch.ones_like(points)), 1)

    def top(self, on_grid=True, n_samples=None):
        points = self._sample1d(horizontal=False, on_grid=on_grid, n_samples=n_samples)
        return torch.stack((torch.zeros_like(points), points), 1)

    def bottom(self, on_grid=True, n_samples=None):
        points = self._sample1d(horizontal=False, on_grid=on_grid, n_samples=n_samples)
        return torch.stack((torch.ones_like(points), points), 1)

    def colloc(self, on_grid=True, n_samples=None, no_boundary=False):
        return self._sample2d(on_grid, n_samples, no_boundary)


if __name__ == '__main__':

    ngrid_h = 10
    ngrid_w = 10

    sampler = SampleSpatial2d(ngrid_h, ngrid_w)
    # print(sampler.refactor)
    # print(sampler.refactor.shape)

    # points = sampler.lhs(n_samples=1000, on_grid=True)
    # print(points)

    points = sampler.right(on_grid=True, n_samples=12)
    # points = sampler.colloc(on_grid=False, n_samples=99, no_boundary=False)
    print(points * sampler.refactor)
    print(points.shape)


