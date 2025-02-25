from manim import *
import numpy as np
import scipy.linalg
from scipy import stats
import itertools
import os

#Make fake 2d data
np.random.seed(123)
X1 = np.random.normal(0, 1, 50)
yhat = X1*0.3 + 1
np.random.seed(1)
eps = np.random.normal(0, 1, 50)
y = yhat + eps
xyvals = np.column_stack((X1, y))

#Make a simple OLS Regression function
# Similar to statsmodels.api.OLS, but I'm using a Docker without that package loaded, so I have to make my own
def myols(x, y, qr = False):
    #normal XTY = XTXb
    #QR: QRTY = QRTQRb or RTY = RTRb

    #bhat ~ N(beta, (XTX)^-1*sigma2)
    #QR: bh ~ N(beta, (QRT*QR)^-1*sigma2) => bh ~ N(beta, RTR^-1*sigma2)
    
    if type(x) == list:
        x = np.array(x)
    X = np.column_stack((np.ones(x.shape[0]), x))
    n = X.shape[0]
    k = X.shape[1]
    out = {"model_matrix": X}
    if qr:
        Q, R = np.linalg.qr(X)
        QTy = np.dot(np.transpose(Q), y)
        Rinv = np.linalg.inv(R)
        bh = np.dot(Rinv, QTy)
    else:
        XTy = np.dot(np.transpose(X), y)
        XTX = np.dot(np.transpose(X), X)
        XTXinv = np.linalg.inv(XTX)
        bh = np.dot(XTXinv, XTy)
    out.update({"coef": bh})
    yhat = np.dot(X, bh)
    out.update({"yhat": yhat})
    resids = np.subtract(y, yhat)
    ssr = np.sum(resids ** 2)
    residvar = np.dot(np.transpose(resids), resids) / (n - k)
    out.update({"resid_var": residvar})
    if qr:
        beta_vcov = residvar * np.linalg.inv(np.dot(np.transpose(R), R))
    else:
        beta_vcov = residvar * XTXinv
    se_beta = np.sqrt(np.diag(beta_vcov))
    out.update({"se_beta": se_beta})
    t_beta = bh/se_beta
    out.update({"tstat": t_beta})
    pvals = 2 * stats.t.sf(np.abs(t_beta), n - k)
    out.update({"pval": pvals})

    return out

def makepermmat(vec, nperms=200):
    vec = np.array(vec)
    perms = []
    for _ in range(nperms):
        permuted_vector = np.random.permutation(vec)
        perms.append(permuted_vector)
    return np.column_stack(perms)

firstreg = myols(X1, y)
perms = makepermmat(y, 50)

def myolspred(myolsobj, newx, as_tuple=False):
    coef = myolsobj["coef"]
    modelmat = np.column_stack((np.ones(newx.shape[0]), newx))
    newpreds = np.dot(modelmat, coef)
    if as_tuple:
        newpreds = tuple(newpreds.tolist())
    return newpreds

def map_float_to_index(value):
    mapping = {
        (float('-inf'), -0.2): 0,
        (-0.20, -0.15): 1,
        (-0.15, -0.10): 2,
        (-0.10, -0.05): 3,
        (-0.05, 0): 4,
        (0, 0.05): 5,
        (0.05, 0.10): 6,
        (0.10, 0.15): 7,
        (0.15, 0.20): 8,
        (0.20, 0.25): 9,
        (0.25, float('inf')): 10,
    }
    
    for (lower, upper), index in mapping.items():
        if lower <= value < upper:
            return index
    return "Undefined"

def have_same_unique_values(arr1, arr2, ordermatters = False):
    unique_arr1 = np.unique(arr1)
    unique_arr2 = np.unique(arr2)

    if ordermatters:
        # Method 1: Using numpy.array_equal (order matters)
        return np.array_equal(unique_arr1, unique_arr2)
    else:
        # Method 2: Using numpy.intersect1d (order doesn't matter)
        intersection = np.intersect1d(unique_arr1, unique_arr2)
        return len(intersection) == len(unique_arr1) and len(intersection) == len(unique_arr2)


class PermTest2D(Scene):
    def construct(self):
        #Title Intro
        title = Text('Visualizing a Simple Permutation Test for\nUnivariable Linear Regression').scale(0.8)
        self.play(Create(title))
        self.wait(2)
        self.play(FadeOut(title))

        #Describe scene
        desctxt = Text("For univariable regression, the outcome y is permuted\nand regression repeatedly run to simulate\nthe coefficient null distribution.").scale(0.6).to_edge(DOWN, buff=0.5)
        
        #initialize histogram
        b1p_bins = np.zeros(11)
        barchart = BarChart(values = b1p_bins.tolist(), 
                           bar_names = ["-0.20", "-0.15", "-0.10", "-0.05", "0", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30"], 
                            y_range=[0, 10, 2], y_length=6, x_length=12, x_axis_config={"font_size": 24},).scale(0.5).to_edge(RIGHT, buff=0.5).shift(UP*2)
        bartitle = MathTex(r"\beta_1^*").next_to(barchart, UP, buff=0.5)

        #Create initial regression
        ax = Axes(x_range=[-4, 3, 1], y_range=[-2, 4, 1]).scale(0.5).to_edge(LEFT, buff=0.5).shift(UP*2)
        self.play(Create(ax), Create(barchart), Create(bartitle), Create(desctxt))
        
        firstb1txt = MathTex(r"True~\beta_1=" + str(round(firstreg["coef"][1], 2))).next_to(ax, UP, buff=1)
        dots = [Dot(ax.c2p(u, v), color=BLUE) for u, v in np.column_stack((X1, y))]
        firstline_ends = [np.array([-4.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])]
        firstline_ys = myolspred(firstreg, np.array([-4, 3]))
        for j in range(2):
            firstline_ends[j][1] = firstline_ys[j]
        firstline = Line(start = ax.c2p(firstline_ends[0][0], firstline_ends[0][1], firstline_ends[0][2]), 
                         end = ax.c2p(firstline_ends[1][0], firstline_ends[1][1], firstline_ends[1][2]), color=YELLOW)
        # firstline = Line(start = ax.c2p(-4, -0.5, 0), end = ax.c2p(3, 2, 0), color=YELLOW)#.to_edge(LEFT, buff=0.5)
        self.play(LaggedStart(*[Write(dot) for dot in dots], lag_ratio=0.01))
        self.wait(1.5)
        self.play(Create(firstline), Create(firstb1txt))
        self.wait(1.5)
        
        #perm regressions
        b1star = []
        self.play(LaggedStart(*[FadeOut(dot) for dot in dots], lag_ratio=0.005), run_time=0.5)
        self.wait(0.5)
        for i in range(50):
            if i>0:
                self.remove(*[dot for dot in tmpdots])
                self.remove(tmpline, firstline, firstb1txt, tmpb1txt, barchart)
            tmpdots = [Dot(ax.c2p(u, v), color=BLUE) for u, v in np.column_stack((X1, perms[i]))]
            tmpreg = myols(X1, perms[i])
            tmpb1txt = MathTex(r"\beta_1^*=" + str(round(tmpreg["coef"][1], 2))).next_to(ax, UP, buff=1)
            b1star.append(tmpreg["coef"][1])
            histbin_index = map_float_to_index(tmpreg["coef"][1]) #list(map(map_float_to_index, values))
            b1p_bins[histbin_index] += 1 #add 1 to a given hist bin
            tmpbarchart = BarChart(values = b1p_bins.tolist(), 
                           bar_names = ["-0.20", "-0.15", "-0.10", "-0.05", "0", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30"], 
                                   y_range=[0, 10, 2], y_length=6, x_length=12, x_axis_config={"font_size": 24},).scale(0.5).to_edge(RIGHT, buff=0.5).shift(UP*2)
            
            permcounter = Text("Permutation #" + str(i))
            tmp_ys = myolspred(tmpreg, np.array([-4, 3]))
            ends = [np.array([-4.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])]
            for j in range(2):
                ends[j][1] = tmp_ys[j]
            tmpline = Line(start = ax.c2p(ends[0][0], ends[0][1], ends[0][2]), 
                         end = ax.c2p(ends[1][0], ends[1][1], ends[1][2]), color=ORANGE)
            self.add(*[dot for dot in tmpdots])
            if i>0:
                self.add(tmpline, tmpb1txt, tmpbarchart)
                self.wait(0.4)
            else: 
                self.play(Transform(firstline, tmpline), Transform(firstb1txt, tmpb1txt), Transform(barchart, tmpbarchart), run_time=1)
                self.wait(1)
        self.wait(2)
        self.play(FadeOut(desctxt))

        #get pval
        ptptxt1 = MathTex(r"Calculate~p:~p=\frac{length(\beta_1^*>True~\beta_1)}{length(\beta_1^*)}").scale(0.8).to_edge(DOWN, buff=0.5)
        tmpbool = [val > firstreg['coef'][1] for val in b1star]
        nb1star_greater = sum(tmpbool)
        finalp = str(round(nb1star_greater/50, 2))
        ptptxt2 = MathTex(r"Calculate~p:~p=\frac{" + str(nb1star_greater) + "}{50}=" + finalp).scale(0.8).to_edge(DOWN, buff=0.5)
        self.play(Create(ptptxt1))
        self.wait(2)
        self.play(Transform(ptptxt1, ptptxt2))
        self.wait(4)

command1 = "manim WQSCalc.py WQSIntro -w -qm"
os.system(command1)