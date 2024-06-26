"""
Unit tests for trust-region optimization routines.

To run it in its simplest form::
  nosetests test_optimize.py

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
                            rosen_hess_prod)
from numpy.testing import (TestCase, assert_, assert_equal, assert_allclose,
                           run_module_suite)


class Accumulator:
    """
    This is for testing callbacks.
    """
    def __init__(self):
        self.count = 0
        self.accum = None

    def __call__(self, x):
        self.count += 1
        if self.accum is None:
            self.accum = np.array(x)
        else:
            self.accum += x


class TestTrustRegionSolvers(TestCase):

    def setUp(self):
        self.x_opt = [1.0, 1.0]
        self.easy_guess = [2.0, 2.0]
        self.hard_guess = [-1.2, 1.0]

    def test_dogleg_accuracy(self):
        # test the accuracy and the return_all option
        x0 = self.hard_guess
        r = minimize(rosen, x0, jac=rosen_der, hess=rosen_hess, tol=1e-8,
                     method='dogleg', options={'return_all': True},)
        assert_allclose(x0, r['allvecs'][0])
        assert_allclose(r['x'], r['allvecs'][-1])
        assert_allclose(r['x'], self.x_opt)

    def test_dogleg_callback(self):
        # test the callback mechanism and the maxiter and return_all options
        accumulator = Accumulator()
        maxiter = 5
        r = minimize(rosen, self.hard_guess, jac=rosen_der, hess=rosen_hess,
                     callback=accumulator, method='dogleg',
                     options={'return_all': True, 'maxiter': maxiter},)
        assert_equal(accumulator.count, maxiter)
        assert_equal(len(r['allvecs']), maxiter+1)
        assert_allclose(r['x'], r['allvecs'][-1])
        assert_allclose(sum(r['allvecs'][1:]), accumulator.accum)

    def test_solver_concordance(self):
        # Assert that dogleg uses fewer iterations than ncg on the Rosenbrock
        # test function, although this does not necessarily mean
        # that dogleg is faster or better than ncg even for this function
        # and especially not for other test functions.
        f = rosen
        g = rosen_der
        h = rosen_hess
        for x0 in (self.easy_guess, self.hard_guess):
            r_dogleg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                method='dogleg', options={'return_all': True})
            r_trust_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                   method='trust-ncg',
                                   options={'return_all': True})
            r_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                             method='newton-cg', options={'return_all': True})
            assert_allclose(self.x_opt, r_dogleg['x'])
            assert_allclose(self.x_opt, r_trust_ncg['x'])
            assert_allclose(self.x_opt, r_ncg['x'])
            assert_(len(r_dogleg['allvecs']) < len(r_ncg['allvecs']))

    def test_trust_ncg_hessp(self):
        for x0 in (self.easy_guess, self.hard_guess):
            r = minimize(rosen, x0, jac=rosen_der, hessp=rosen_hess_prod,
                         tol=1e-8, method='trust-ncg')
            assert_allclose(self.x_opt, r['x'])


if __name__ == '__main__':
    run_module_suite()
