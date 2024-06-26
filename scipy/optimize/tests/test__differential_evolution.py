"""
Unit tests for the differential global minimization algorithm.
"""
from scipy.optimize import _differentialevolution
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.optimize import differential_evolution
import numpy as np
from scipy.optimize import rosen
from numpy.testing import (assert_equal, TestCase, assert_allclose,
                           run_module_suite, assert_almost_equal,
                           assert_string_equal)


class TestDifferentialEvolutionSolver(TestCase):

    def setUp(self):
        self.old_seterr = np.seterr(invalid='raise')
        self.limits = np.array([[0., 0.],
                                [2., 2.]])
        self.bounds = [(0., 2.), (0., 2.)]

        self.dummy_solver = DifferentialEvolutionSolver(self.quadratic,
                                                        [(0, 100)])

        #dummy_solver2 will be used to test mutation strategies
        self.dummy_solver2 = DifferentialEvolutionSolver(self.quadratic,
                                                         [(0, 1)],
                                                         popsize=7,
                                                         mutation=0.5)
        #create a population that's only 7 members long
        #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        population = np.atleast_2d(np.arange(0.1, 0.8, 0.1)).T
        self.dummy_solver2.population = population

    def tearDown(self):
        np.seterr(**self.old_seterr)

    def quadratic(self, x):
        return x[0]**2

    def test__strategy_resolves(self):
        #test that the correct mutation function is resolved by
        #different requested strategy arguments
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='best1exp')
        assert_equal(solver.strategy, 'best1exp')
        assert_equal(solver.mutation_func.__name__, '_best1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='best1bin')
        assert_equal(solver.strategy, 'best1bin')
        assert_equal(solver.mutation_func.__name__, '_best1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand1bin')
        assert_equal(solver.strategy, 'rand1bin')
        assert_equal(solver.mutation_func.__name__, '_rand1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand1exp')
        assert_equal(solver.strategy, 'rand1exp')
        assert_equal(solver.mutation_func.__name__, '_rand1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand2exp')
        assert_equal(solver.strategy, 'rand2exp')
        assert_equal(solver.mutation_func.__name__, '_rand2')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='best2bin')
        assert_equal(solver.strategy, 'best2bin')
        assert_equal(solver.mutation_func.__name__, '_best2')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand2bin')
        assert_equal(solver.strategy, 'rand2bin')
        assert_equal(solver.mutation_func.__name__, '_rand2')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand2exp')
        assert_equal(solver.strategy, 'rand2exp')
        assert_equal(solver.mutation_func.__name__, '_rand2')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='randtobest1bin')
        assert_equal(solver.strategy, 'randtobest1bin')
        assert_equal(solver.mutation_func.__name__, '_randtobest1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='randtobest1exp')
        assert_equal(solver.strategy, 'randtobest1exp')
        assert_equal(solver.mutation_func.__name__, '_randtobest1')

    def test__mutate1(self):
        #strategies */1/*, i.e. rand/1/bin, best/1/exp, etc.
        result = np.array([0.05])
        trial = self.dummy_solver2._best1((2, 3, 4, 5, 6))
        assert_allclose(trial, result)

        result = np.array([0.25])
        trial = self.dummy_solver2._rand1((2, 3, 4, 5, 6))
        assert_allclose(trial, result)

    def test__mutate2(self):
        #strategies */2/*, i.e. rand/2/bin, best/2/exp, etc.
        #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        result = np.array([-0.1])
        trial = self.dummy_solver2._best2((2, 3, 4, 5, 6))
        assert_allclose(trial, result)

        result = np.array([0.1])
        trial = self.dummy_solver2._rand2((2, 3, 4, 5, 6))
        assert_allclose(trial, result)

    def test__randtobest1(self):
        #strategies randtobest/1/*
        result = np.array([0.1])
        trial = self.dummy_solver2._randtobest1(1, (2, 3, 4, 5, 6))
        assert_allclose(trial, result)

    def test_can_init_with_dithering(self):
        mutation = (0.5, 1)
        solver = DifferentialEvolutionSolver(self.quadratic,
                                             self.bounds,
                                             mutation=mutation)

        self.assertEqual(solver.dither, list(mutation))

    def test_invalid_mutation_values_arent_accepted(self):
        func = rosen
        mutation = (0.5, 3)
        self.assertRaises(ValueError,
                          DifferentialEvolutionSolver,
                          func,
                          self.bounds,
                          mutation=mutation)

        mutation = (-1, 1)
        self.assertRaises(ValueError,
                          DifferentialEvolutionSolver,
                          func,
                          self.bounds,
                          mutation=mutation)

        mutation = (0.1, np.nan)
        self.assertRaises(ValueError,
                          DifferentialEvolutionSolver,
                          func,
                          self.bounds,
                          mutation=mutation)

        mutation = (0.5)
        solver = DifferentialEvolutionSolver(func,
                                             self.bounds,
                                             mutation=mutation)
        assert_equal(0.5, solver.scale)
        assert_equal(None, solver.dither)

    def test__scale_parameters(self):
        trial = np.array([0.3])
        assert_equal(30, self.dummy_solver._scale_parameters(trial))

        # it should also work with the limits reversed
        self.dummy_solver.limits = np.array([[100], [0.]])
        assert_equal(30, self.dummy_solver._scale_parameters(trial))

    def test__unscale_parameters(self):
        trial = np.array([30])
        assert_equal(0.3, self.dummy_solver._unscale_parameters(trial))

        # it should also work with the limits reversed
        self.dummy_solver.limits = np.array([[100], [0.]])
        assert_equal(0.3, self.dummy_solver._unscale_parameters(trial))

    def test__ensure_constraint(self):
        trial = np.array([1.1, -100, 2., 300., -0.00001])
        self.dummy_solver._ensure_constraint(trial)
        assert_equal(np.all(trial <= 1), True)

    def test_differential_evolution(self):
        # test that the Jmin of DifferentialEvolutionSolver
        # is the same as the function evaluation
        solver = DifferentialEvolutionSolver(self.quadratic, [(-2, 2)])
        result = solver.solve()
        assert_almost_equal(result.fun, self.quadratic(result.x))

    def test_best_solution_retrieval(self):
        # test that the getter property method for the best solution works.
        solver = DifferentialEvolutionSolver(self.quadratic, [(-2, 2)])
        result = solver.solve()
        assert_equal(result.x, solver.x)

    def test_callback_terminates(self):
        # test that if the callback returns true, then the minimization halts
        bounds = [(0, 2), (0, 2)]

        def callback(param, convergence=0.):
            return True

        result = differential_evolution(rosen, bounds, callback=callback)

        assert_string_equal(result.message,
                                'callback function requested stop early '
                                'by returning True')

    def test_args_tuple_is_passed(self):
        # test that the args tuple is passed to the cost function properly.
        bounds = [(-10, 10)]
        args = (1., 2., 3.)

        def quadratic(x, *args):
            if type(args) != tuple:
                raise ValueError('args should be a tuple')
            return args[0] + args[1] * x + args[2] * x**2.

        result = differential_evolution(quadratic,
                                        bounds,
                                        args=args,
                                        polish=True)
        assert_almost_equal(result.fun, 2 / 3.)

    def test_init_with_invalid_strategy(self):
        #test that passing an invalid strategy raises ValueError
        func = rosen
        bounds = [(-3, 3)]
        self.assertRaises(ValueError,
                          differential_evolution,
                          func,
                          bounds,
                          strategy='abc')

    def test_bounds_checking(self):
        #test that the bounds checking works
        func = rosen
        bounds = [(-3, None)]
        self.assertRaises(ValueError,
                          differential_evolution,
                          func,
                          bounds)
        bounds = [(-3)]
        self.assertRaises(ValueError,
                          differential_evolution,
                          func,
                          bounds)
        bounds = [(-3, 3), (3, 4, 5)]
        self.assertRaises(ValueError,
                          differential_evolution,
                          func,
                          bounds)

    def test_select_samples(self):
        #select_samples should return 5 separate random numbers.
        limits = np.arange(12., dtype='float64').reshape(2, 6)
        bounds = list(zip(limits[0, :], limits[1, :]))
        solver = DifferentialEvolutionSolver(None, bounds, popsize=1)
        candidate = 0
        r1, r2, r3, r4, r5 = solver._select_samples(candidate, 5)
        assert_equal(
            len(np.unique(np.array([candidate, r1, r2, r3, r4, r5]))), 6)

    def test_maxiter_stops_solve(self):
        #test that if the maximum number of iterations is exceeded
        #the solver stops.
        solver = DifferentialEvolutionSolver(rosen, self.bounds, maxiter=1)
        result = solver.solve()
        assert_equal(result.success, False)
        assert_equal(result.message,
                        'Maximum number of iterations has been exceeded.')

    def test_maxfun_stops_solve(self):
        #test that if the maximum number of function evaluations is exceeded
        #during initialisation the solver stops
        solver = DifferentialEvolutionSolver(rosen, self.bounds, maxfun=1)
        result = solver.solve()

        assert_equal(result.nfev, 2)
        assert_equal(result.success, False)
        assert_equal(result.message,
                         'Maximum number of function evaluations has '
                              'been exceeded.')

        #test that if the maximum number of function evaluations is exceeded
        #during the actual minimisation, then the solver stops.
        #Have to turn polishing off, as this will still occur even if maxfun
        #is reached. For popsize=5 and len(bounds)=2, then there are only 10
        #function evaluations during initialisation.
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             popsize=5,
                                             polish=False,
                                             maxfun=40)
        result = solver.solve()

        assert_equal(result.nfev, 41)
        assert_equal(result.success, False)
        assert_equal(result.message,
                         'Maximum number of function evaluations has '
                              'been exceeded.')

    def test_quadratic(self):
        # test the quadratic function from object
        solver = DifferentialEvolutionSolver(self.quadratic,
                                             [(-100, 100)],
                                             tol=0.02)
        solver.solve()

    def test_quadratic_from_diff_ev(self):
        # test the quadratic function from differential_evolution function
        differential_evolution(self.quadratic,
                               [(-100, 100)],
                               tol=0.02)

    def test_seed_gives_repeatability(self):
        result = differential_evolution(self.quadratic,
                                        [(-100, 100)],
                                        polish=False,
                                        seed=1,
                                        tol=0.5)
        result2 = differential_evolution(self.quadratic,
                                        [(-100, 100)],
                                        polish=False,
                                        seed=1,
                                        tol=0.5)
        assert_equal(result.x, result2.x)

    def test_exp_runs(self):
        # test whether exponential mutation loop runs
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='best1exp',
                                             maxiter=1)

        solver.solve()

    def test__make_random_gen(self):
        # If seed is None, return the RandomState singleton used by np.random.
        # If seed is an int, return a new RandomState instance seeded with seed.
        # If seed is already a RandomState instance, return it.
        # Otherwise raise ValueError.
        rsi = _differentialevolution._make_random_gen(1)
        assert_equal(type(rsi), np.random.RandomState)
        rsi = _differentialevolution._make_random_gen(rsi)
        assert_equal(type(rsi), np.random.RandomState)
        rsi = _differentialevolution._make_random_gen(None)
        assert_equal(type(rsi), np.random.RandomState)
        self.assertRaises(
            ValueError, _differentialevolution._make_random_gen, 'a')


if __name__ == '__main__':
    run_module_suite()
