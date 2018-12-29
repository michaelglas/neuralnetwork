'''
Created on 26.12.2018

@author: michi
'''
import unittest
from neuralnetwork import ACT_FUNCS
import math
import numpy


class Test(unittest.TestCase):


    def testSigmoid(self):
        
        s = ACT_FUNCS["sigmoid"]
        
        r = (1/(1+math.exp(10)),1/(1+math.exp(1)),1/2,1/(1+math.exp(-1)),1/(1+math.exp(-10)))
        
        self.assertAlmostEqual(r[0],s(-10),15)
        self.assertAlmostEqual(r[1],s(-1),15)
        self.assertEqual(r[2],s(0))
        self.assertAlmostEqual(r[3],s(1),15)
        self.assertAlmostEqual(r[4],s(10),15)
        
        dr = [x * (1-x) for x in r]
        
        # sig(x) * (1-sig(x)) = 1/(1+e^-x) * (1-1/(1+e^-x)) =
        #    1/(1+e^-x) * (-e^-x/(1+e^-x))
        self.assertAlmostEqual((1-1/(1+math.exp(1)))/(1+math.exp(1)),s.derivative(-1),15)
        self.assertAlmostEqual((1-1/(1+math.exp(10)))/(1+math.exp(10)),s.derivative(-10),15)
        self.assertEqual(0.25,s.derivative(0))
        self.assertAlmostEqual((1-1/(1+math.exp(-1)))/(1+math.exp(-1)),s.derivative(1),15)
        self.assertAlmostEqual((1-1/(1+math.exp(-10)))/(1+math.exp(-10)),s.derivative(10),15)
        
        m = numpy.matrix([-10,-1,0,1,10])
        
        sm = s(m)
        
        print ("sm = %s"%(sm,))

        for i in range(len(r)):
            self.assertAlmostEqual(r[i],sm[0,i],15)
            
        dm = s.derivative(m)
        
        print ("dm = %s"%(dm,))

        for i in range(len(dr)):
            self.assertAlmostEqual(dr[i],dm[0,i],15)
        
        
    def testIdentity(self):
        
        s = ACT_FUNCS["identity"]
        
        self.assertEqual(-1,s(-1))
        self.assertEqual(-10,s(-10))
        self.assertEqual(0,s(0))
        self.assertEqual(1,s(1))
        self.assertEqual(10,s(10))

        self.assertEqual(1,s.derivative(-1))
        self.assertEqual(1,s.derivative(-10))
        self.assertEqual(1,s.derivative(0))
        self.assertEqual(1,s.derivative(1))
        self.assertEqual(1,s.derivative(10))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testSigmoid']
    unittest.main()