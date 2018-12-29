'''
Created on 26.12.2018

@author: michi
'''
import unittest
from neuralnetwork import layeredneuralnetwork, ACT_FUNCS
import numpy


class Test(unittest.TestCase):


    def testSimpleIdentity(self):
        
        m1 = numpy.matrix([[1.0,2.0,3.0],[3.0,4.0,5.0]])
        m2 = numpy.matrix([[7.0,8.0,9.0]]).T
        
        network = layeredneuralnetwork.from_values([2,3,1],ACT_FUNCS["identity"],
                                                   [m1,m2])
        
        l_1_1 = 5 * 1 + 7*3
        l_1_2 = 5 * 2 + 7*4
        l_1_3 = 5 * 3 + 7*5
        
        v_test = l_1_1 * 7+ l_1_2 * 8 + l_1_3 * 9
        
        v = network.calculate(numpy.matrix([5,7]))
        
        print ("v,v_test=%s,%s"%(v,v_test))
        
        self.assertEqual(v_test,v)

        inp = numpy.matrix([[5,7],[2.5,3.5]])
        
        v2 = network.calculate(inp)
        
        print ("v2=%s"%(v2,))
        
        real_output = numpy.matrix([v_test,v_test*0.5]).T
        
        self.assertTrue((real_output==v2).all())
        
        self.assertEqual(0,network.calculate_error(inp,real_output))
        
        expected_output = numpy.matrix([v_test+1,v_test*0.5+2]).T
        
        d = network.calculate_error(inp,expected_output)

        print ("d=%s"%(d,))

        self.assertEqual(2.5,d)

        deltas = network.calculate_deltas(inp,expected_output)

        print ("weights=%s"%(network.weights,))
        
        print ("deltas=%s"%(deltas,))
        
        dx = 1.0e-8
        
        for l in range(0,2):
            
            shp = deltas[l].shape
            
            for i in range(0,shp[0]):
                for j in range(0,shp[1]):
                    nw = [ m1.copy(),m2.copy() ]
                    
                    x = nw[l][i,j]
                    
                    nw[l][i,j] = x * (1.0 + dx)
                    
                    network2 = layeredneuralnetwork.from_values([2,3,1],
                                                ACT_FUNCS["identity"],
                                                nw)
            
                    d2 = network2.calculate_error(inp,expected_output)
                    
                    ddelta = (d2-d)/(x*dx)
                    
                    print ("l,i,j,delta,ddelta = %s,%s,%s,%s,%.8g"%(l,i,j,deltas[l][i,j],ddelta))
        
                    self.assertAlmostEqual(deltas[l][i,j],ddelta,3)

    def testSimpleSigmoid(self):
        
        s = ACT_FUNCS["sigmoid"]
        
        m1 = numpy.matrix([[0.1,0.2,0.3],[0.3,0.4,0.5]])
        
        m2 = numpy.matrix([[0.7,0.8,0.9]]).T
        
        network = layeredneuralnetwork.from_values([2,3,1],s,[m1,m2])
        
        l_1_1 = s(0.5 * 0.1 + 0.7*0.3)
        l_1_2 = s(0.5 * 0.2 + 0.7*0.4)
        l_1_3 = s(0.5 * 0.3 + 0.7*0.5)
        
        v_test = s(l_1_1 * 0.7+ l_1_2 * 0.8 + l_1_3 * 0.9)
        
        v = network.calculate(numpy.matrix([0.5,0.7]))
        
        print ("v,v_test=%s,%s"%(v,v_test))
        
        self.assertEqual(v_test,v)

        l_1_1 = s(0.25 * 0.1 + 0.35*0.3)
        l_1_2 = s(0.25 * 0.2 + 0.35*0.4)
        l_1_3 = s(0.25 * 0.3 + 0.35*0.5)
        
        v2_test = s(l_1_1 * 0.7+ l_1_2 * 0.8 + l_1_3 * 0.9)

        inp = numpy.matrix([[0.5,0.7],[0.25,0.35]])

        v2 = network.calculate(inp)
        
        print ("v2=%s"%(v2,))
        
        expected_output = numpy.matrix([v_test,v2_test]).T
        
        self.assertTrue((expected_output==v2).all())
        
        d = network.calculate_error(inp, expected_output)
        
        self.assertEqual(0,d)
        
        inp = numpy.matrix([[0.51,0.71],[0.26,0.36]])

        d = network.calculate_error(inp, expected_output)
        
        print ("d=%s"%(d,))
        
        deltas = network.calculate_deltas(inp, expected_output)
        
        print ("weights=%s"%(network.weights,))
        
        print ("deltas=%s"%(deltas,))
        
        dx = 1.0e-8
        
        for l in range(0,2):
            
            shp = deltas[l].shape
            
            for i in range(0,shp[0]):
                for j in range(0,shp[1]):
                    
                    nw = [ m1.copy(),m2.copy() ]
                    
                    x = nw[l][i,j]
                    
                    nw[l][i,j] = x * (1.0 + dx)
                    
                    network2 = layeredneuralnetwork.from_values([2,3,1],s,nw)
            
                    d2 = network2.calculate_error(inp,expected_output)
                    
                    ddelta = (d2-d)/(x*dx)
                    
                    print ("l,i,j,delta,ddelta = %s,%s,%s,%s,%.8g"%(l,i,j,deltas[l][i,j],ddelta))
        
                    self.assertAlmostEqual(deltas[l][i,j],ddelta,3)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testSimple']
    unittest.main()