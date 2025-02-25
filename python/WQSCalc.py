from manim import *
import os

def list_subtract(A,B):
     if isinstance(A,list):
         return [list_subtract(ra,rb) for ra,rb in zip(A,B) ]
     else:
         return A-B
     
def value_matrix(value, nrow, ncol):
    matrix = [[value for _ in range(ncol)] for _ in range(nrow)]
    return matrix

def make_bootstrap_matrices(mat, nboots=100):
    if isinstance(mat, list):
        mat = np.array(mat)
    n_rows = mat.shape[0]
    bootmats = []
    for _ in range(nboots):
        random_ind = np.random.choice(n_rows, size=n_rows, replace=True)
        tempbootmat = mat[random_ind]
        bootmats.append(tempbootmat)
    return bootmats

class WQSIntro(Scene):
    def construct(self):
        #Title Intro
        title = Text('Linear Additive Effects\nof Exposure Mixtures:\n\nWeighted Quantile Sum (WQS)\nRegression').scale(0.8)
        self.play(Create(title))
        self.wait(2)
        self.play(FadeOut(title))
        subtitle1 = Text("Part 1: The Model Structure")
        self.play(Create(subtitle1))
        self.wait(1)
        self.play(FadeOut(subtitle1))
        
        # Define the matrix and vector
        matrix_pre = [[1.001, 1.045], [0.002, 3.142], [0.056, 2.017]]
        matrix = [[2, 0], [0, 2], [1, 1]]
        vector = [0.8, 0.2]

        # Create Manim objects for the matrix and vector
        matrixpre_mobject = Matrix(matrix_pre)
        matrixpre_mobject.set_column_colors(RED, BLUE)
        matrix_mobject = Matrix(matrix)
        matrix_mobject.set_column_colors(RED, BLUE)
        vector_mobject = Matrix([[vector[0]], [vector[1]]])
        vector_mobject.set_row_colors(RED, BLUE)
        
        # Position the matrix and vector
        matrixpre_mobject.to_corner(UP + LEFT, buff=1.5)
        matrix_mobject.to_corner(UP + LEFT, buff=1.5)
        vector_mobject.next_to(matrix_mobject, RIGHT, buff=2)

        # Name the matrix and vector X and w
        matrixpre_name = MathTex("X").next_to(matrixpre_mobject, UP, buff=0.5)
        matrix_name = MathTex("X_q").next_to(matrix_mobject, UP, buff=0.5)
        vname = MathTex("w").next_to(vector_mobject, UP, buff=0.5)

        #Text
        qttext = Text('Quantile\nTransformation', stroke_width=0).next_to(matrixpre_mobject, RIGHT, buff=2)
        emtext = Text('Mixture\nExposure\nMatrix', stroke_width=0).next_to(matrixpre_mobject, RIGHT, buff=2)
        wtext = Text('Exposure\nWeights', stroke_width=0).next_to(vector_mobject, RIGHT, buff=2)
        row2text = Text('P2', font_size=20).next_to(matrixpre_mobject, LEFT, buff=0.1)
        row1text = Text('P1', font_size=20).next_to(row2text, UP, buff=0.6)
        row3text = Text('P3', font_size=20).next_to(row2text, DOWN, buff=0.6)
        col1texta = Text('E1', font_size=20, color=RED).next_to(matrixpre_mobject, UP, buff=0.1).shift(LEFT*0.71)
        col2texta = Text('E2', font_size=20, color=BLUE).next_to(matrixpre_mobject, UP, buff=0.1).shift(RIGHT*0.71)
        col1textb = Text('E1', font_size=20, color=RED).next_to(matrix_mobject, UP, buff=0.1).shift(LEFT*0.65)
        col2textb = Text('E2', font_size=20, color=BLUE).next_to(matrix_mobject, UP, buff=0.1).shift(RIGHT*0.65)
        vectext1 = Text('E1', font_size=20, color=RED).next_to(vector_mobject, LEFT, buff=0.1).shift(UP*0.4)
        vectext2 = Text('E2', font_size=20, color=BLUE).next_to(vector_mobject, LEFT, buff=0.1).shift(DOWN*0.4)
        
        # X -> Xq
        self.play(Create(emtext))
        self.play(Create(matrixpre_mobject), Create(matrixpre_name))
        self.wait(0.5)
        self.play(Create(row1text), Create(row2text), Create(row3text), 
                 Create(col1texta), Create(col2texta))
        self.play(FadeOut(emtext))
        self.remove(emtext)
        self.wait(0.5)
        self.play(Create(qttext))
        self.play(Transform(matrixpre_mobject, matrix_mobject), Transform(matrixpre_name, matrix_name), 
                 Transform(col1texta, col1textb), Transform(col2texta, col2textb))
        self.wait(0.5)
        self.play(FadeOut(qttext))

        # Bring in w
        dotsign = MathTex("\dot").next_to(matrix_mobject, RIGHT, buff=1)
        self.play(Create(wtext))
        wtvec = VGroup(vector_mobject,vname,vectext1,vectext2)
        self.play(Create(wtvec))
        self.play(Create(dotsign))
        self.play(FadeOut(wtext))
        self.wait(1)
        
        # Bring in equals sign and result placeholder
        equals_sign = MathTex("=").next_to(vector_mobject, RIGHT, buff=1)
        result_placeholder = Matrix([["?"], ["?"], ["?"]]).next_to(equals_sign, RIGHT, buff=1)
        wqsname = MathTex("WQS").next_to(result_placeholder, UP, buff=0.5)
        row2text_wqs = Text('P2', font_size=20).next_to(result_placeholder, LEFT, buff=0.1)
        row1text_wqs = Text('P1', font_size=20).next_to(row2text_wqs, UP, buff=0.6)
        row3text_wqs = Text('P3', font_size=20).next_to(row2text_wqs, DOWN, buff=0.6)
        self.play(Create(equals_sign), Create(result_placeholder))
        self.play(Create(row1text_wqs), Create(row2text_wqs), Create(row3text_wqs), run_time=0.25)
        self.play(Create(wqsname))

        # Calculate the dot product step-by-step
        result = [
            round(matrix[0][0] * vector[0] + matrix[0][1] * vector[1], 1),
            round(matrix[1][0] * vector[0] + matrix[1][1] * vector[1], 1),
            round(matrix[2][0] * vector[0] + matrix[2][1] * vector[1], 1)
        ]

        # Animate the calculation
        for i in range(3):
            # Highlight the relevant row and vector elements
            row = matrix_mobject.get_rows()[i]
            self.play(
                Indicate(row),
                Indicate(vector_mobject),
            )
            
            # Write the intermediate calculation
            calculation_text = MathTex(
                f"({matrix[i][0]} \\times {vector[0]}) + ({matrix[i][1]} \\times {vector[1]})"
            ).next_to(result_placeholder, DOWN, buff=1.5)
            if i>0:
                self.play(Write(calculation_text, run_time=1))
            else:
                self.play(Write(calculation_text))
                self.wait(1)
            
            # Calculate and display the result for the row
            calculated_value = matrix[i][0] * vector[0] + matrix[i][1] * vector[1]
            if i>0:
                self.play(Transform(calculation_text, DecimalNumber(
                    calculated_value, num_decimal_places=1).next_to(
                    result_placeholder, DOWN, buff=1.5), run_time=0.5))
                self.wait(0.1)
            else:
                self.play(Transform(calculation_text, DecimalNumber(
                    calculated_value, num_decimal_places=1).next_to(
                    result_placeholder, DOWN, buff=1.5)))
                self.wait(0.25)
            
            # Update the result matrix
            newmat = [["?"], ["?"], ["?"]]
            for j in range(i+1):
                newmat[j][0] = result[j]
            new_result = Matrix(newmat).next_to(equals_sign, RIGHT, buff=1)
            new_result.set_column_colors(PURPLE)
            if i>0:
                self.play(Transform(result_placeholder, new_result), FadeOut(calculation_text), run_time=0.5)
                self.wait(0.1)
            else: 
                self.play(wqsname.animate.next_to(new_result, UP, buff=0.5), 
                         Transform(result_placeholder, new_result))
                self.play(FadeOut(calculation_text))
                self.wait(0.25)
        
        self.wait(1)

        #Clear the screen
        EndScene1 = VGroup(row2text,row1text,row3text,dotsign,equals_sign,wtvec,col2textb,col1textb,matrix_mobject,
                          matrix_name,result_placeholder,col2texta,col1texta,matrixpre_mobject,matrixpre_name) 
        wqsgroup = VGroup(new_result,row2text_wqs,row3text_wqs,row1text_wqs,wqsname)
        self.play(FadeOut(EndScene1))
        # self.play(*[FadeOut(mob, run_time=0.5) for mob in self.mobjects])
        
        #WQS*Beta
        self.play(wqsgroup.animate.to_corner(DOWN + LEFT, buff=1.5))
        self.wait(1)
        
        #Explain beta1/beta0
        astsign = MathTex("*").next_to(new_result, RIGHT, buff=1)
        b1expltext = Text('Overall\nMixture\nAssociation', stroke_width=0).next_to(new_result, RIGHT, buff=4)
        b0expltext = Text('Intercept', stroke_width=0).next_to(new_result, RIGHT, buff=6)
        beta1 = MathTex("0.5").next_to(new_result, RIGHT, buff=2)
        plussign = MathTex("+").next_to(beta1, RIGHT, buff=1)
        beta0 = MathTex("2").next_to(beta1, RIGHT, buff=2)
        b1text = Tex(r"$\beta_1$").next_to(beta1, UP, buff=0.6) 
        b0text = Tex(r"$\beta_0$").next_to(beta0, UP, buff=0.6)
        self.play(Create(b1expltext, run_time=0.5))
        self.wait(0.25)
        self.play(Create(beta1, run_time=0.5), Create(b1text, run_time=0.5), Create(astsign, run_time=0.5))
        self.wait(0.25)
        self.play(FadeOut(b1expltext, run_time=0.5))
        self.play(Create(b0expltext, run_time=0.5))
        self.wait(0.25)
        self.play(Create(beta0, run_time=0.5), Create(b0text, run_time=0.5), Create(plussign, run_time=0.5))
        self.play(FadeOut(b0expltext, run_time=0.5))

        #Bring in yhat
        equals_sign = MathTex("=").next_to(beta0, RIGHT, buff=1)
        ytext = MathTex("E[y]").next_to(beta0, RIGHT, buff=2)
        wqsv = [1.6, 0.4, 1.0]
        ey = [i * 0.5 + 2 for i in wqsv]
        yhat = Matrix([[ey[0]], [ey[1]], [ey[2]]]).next_to(beta0, RIGHT, buff=2)
        row2text_y = Text('P2', font_size=20).next_to(yhat, LEFT, buff=0.1)
        row1text_y = Text('P1', font_size=20).next_to(row2text_y, UP, buff=0.6)
        row3text_y = Text('P3', font_size=20).next_to(row2text_y, DOWN, buff=0.6)
        yhattext = Text("Predicted\nOutcome").next_to(ytext, UP, buff=1)
        self.play(Create(equals_sign, run_time=0.5), Create(ytext, run_time=0.5), Create(yhattext))
        self.wait(1)
        self.play(ytext.animate.next_to(yhat, UP, buff=0.5), FadeOut(yhattext))
        self.remove(yhattext)
        yhatgroup = VGroup(yhat, row1text_y, row2text_y, row3text_y)
        self.play(Create(yhatgroup), run_time=0.5)
        self.wait(2)

        #Resids
        yhatgroup2 = VGroup(yhat, row1text_y, row2text_y, row3text_y, ytext)
        ytrue = Matrix([[3.8],[1.9],[2.5]]).to_corner(DOWN + LEFT, buff=1.5)
        ytname = MathTex("y_{obs}").next_to(ytrue, UP, buff=0.5)
        yttext = Text("Observed\nOutcome").next_to(ytrue, UP, buff=1.25)
        row2text_yt = Text('P2', font_size=20).next_to(ytrue, LEFT, buff=0.1)
        row1text_yt = Text('P1', font_size=20).next_to(row2text_yt, UP, buff=0.6)
        row3text_yt = Text('P3', font_size=20).next_to(row2text_yt, DOWN, buff=0.6)
        ytgroup = VGroup(ytrue,ytname,row1text_yt,row2text_yt,row3text_yt)
        minus_sign = MathTex("-").next_to(ytrue, RIGHT, buff=1)
        equals_sign2 = MathTex("=").next_to(minus_sign, RIGHT, buff=3.5)
        eps = list_subtract([3.8,1.9,2.5], ey)
        eps = [round(i,1) for i in eps]
        resid = Matrix([[eps[0]],[eps[1]],[eps[2]]]).next_to(equals_sign2, RIGHT, buff=1)
        epsname = Tex("$\epsilon$").next_to(resid, UP, buff=0.5)
        residtext = Text("Residuals").next_to(resid, UP, buff=1)
        self.play(FadeOut(VGroup(equals_sign, beta0, beta1, b0text, b1text, plussign, astsign, wqsgroup, new_result)))
        self.play(Create(ytgroup), Create(yttext))
        self.wait(0.5)
        self.play(FadeOut(yttext),yhatgroup2.animate.next_to(ytgroup, RIGHT, buff=2), Create(minus_sign))
        self.wait(0.5)
        self.play(Create(equals_sign2), Create(resid), Create(epsname), Create(residtext))
        self.wait(0.5)

        #square resids
        eps_sq = [round(i ** 2,1) for i in eps]
        resid_sq = Matrix([[eps_sq[0]],[eps_sq[1]],[eps_sq[2]]]).next_to(equals_sign2, RIGHT, buff=1)
        epsname2 = Tex("$\epsilon^2$").next_to(resid_sq, UP, buff=0.5)
        residtext2 = Text("Squared\nResiduals").next_to(resid_sq, UP, buff=1.25)
        self.play(FadeOut(VGroup(equals_sign2, minus_sign, ytgroup, yhatgroup2)), Transform(residtext, residtext2), Transform(epsname, epsname2), Transform(resid, resid_sq))
        self.wait(1)

        #sum loss
        rss = Matrix([[1.1]]).next_to(equals_sign2, RIGHT, buff=1)
        rsstext = Text("Sum of\nSquared\nResiduals").next_to(rss, UP, buff=1.5)
        self.play(FadeOut(VGroup(residtext,epsname2, epsname)), Transform(residtext2, rsstext))#, Create(rsstext))
        self.play(Transform(resid_sq, rss), FadeOut(resid))
        self.wait(1)
        losstext = Text("This is the loss\nto be minimized").next_to(rss, UP, buff=1.5)
        self.play(FadeOut(residtext2), Transform(rsstext, losstext))
        self.wait(2)
        

class WQSSteps(Scene):
    def construct(self):
        #Title Intro
        title = Text('Linear Additive Effects\nof Exposure Mixtures:\n\nWeighted Quantile Sum (WQS)\nRegression').scale(0.8)
        self.play(Create(title))
        self.wait(2)
        self.play(FadeOut(title))
        subtitle1 = Text("Part 2: Explaining the Two Stage\nWQS Regression Algorithm:\n\n1. Training Weights and 2. GLM for Coefficients").scale(0.8)
        self.play(Create(subtitle1))
        self.wait(2)
        self.play(FadeOut(subtitle1))

        #Xq
        matrix = [[2, 0], [0, 2], [1, 1]]
        matrix_mobject = Matrix(matrix).set_column_colors(RED, BLUE).to_corner(UP + LEFT, buff=1.5)
        row2text = Text('P2', font_size=20).next_to(matrix_mobject, LEFT, buff=0.1)
        row1text = Text('P1', font_size=20).next_to(row2text, UP, buff=0.6)
        row3text = Text('P3', font_size=20).next_to(row2text, DOWN, buff=0.6)
        col1text = Text('E1', font_size=20, color=RED).next_to(matrix_mobject, UP, buff=0.1).shift(LEFT*0.65)
        col2text = Text('E2', font_size=20, color=BLUE).next_to(matrix_mobject, UP, buff=0.1).shift(RIGHT*0.65)
        matrix_name = MathTex("X_q").next_to(matrix_mobject, UP, buff=0.5)
        matgroup = VGroup(matrix_mobject, row1text, row2text, row3text, col1text, col2text, matrix_name)
        
        #Training
        trtext = Text("Stage 1: Training Weights by Bootstrapping\nXq and BFGS Optimization").scale(0.7).to_edge(UP)
        bresinit = Matrix(value_matrix("?", 5, 5)).to_edge(RIGHT, buff=1).shift(DOWN*1).set_column_colors(WHITE, RED, BLUE, WHITE, WHITE)
        c1t = Text('Boot #', font_size=20).next_to(bresinit, UP, buff=0.1).shift(LEFT*2.5)
        c2t = MathTex('w_{E1}', font_size=25, color=RED).next_to(bresinit, UP, buff=0.1).shift(LEFT*1.5)
        c3t = MathTex('w_{E2}', font_size=25, color=BLUE).next_to(bresinit, UP, buff=0.1).shift(LEFT*0.2)
        c4t = Tex(r"$\beta_1^*$", font_size=25).next_to(bresinit, UP, buff=0.1).shift(RIGHT*1.1)
        c5t = Text('t-stat', font_size=20).next_to(bresinit, UP, buff=0.1).shift(RIGHT*2.5)
        bresgroup = VGroup(bresinit, c1t, c2t, c3t, c4t, c5t)
        self.play(Create(trtext), Create(matrix_mobject.shift(DOWN*2)), Create(matrix_name.shift(DOWN*2)), Create(bresgroup))
        self.wait(2)
        
        #animating bootstrapping
        bresfull = [
            [1,0.6,0.4,0.4,2.3],
            [2,0.1,0.9,-0.1,0.5],
            [3,0.8,0.2,0.5,2.8],
            [4,0.9,0.1,0.6,3.1],
            [5,0.8,0.2,0.5,2.8]
        ]
        for i in range(5):
            tmpmat = Matrix(make_bootstrap_matrices(matrix, 1)[0]).to_corner(UP + LEFT, buff=1.5).shift(DOWN*2).set_column_colors(RED, BLUE)
            tmpmatname = Text("Bootstrap Xq #" + str(int(i+1))).next_to(tmpmat, UP, buff=0.5).scale(0.7)
            tmpbres = value_matrix("?", 5, 5)
            for j in range(i+1):
                tmpbres[j] = bresfull[j]
            tmpbreso = Matrix(tmpbres).to_edge(RIGHT, buff=1).shift(DOWN*1).set_column_colors(WHITE, RED, BLUE, WHITE, WHITE)
            if i==0:
                self.play(Transform(matrix_mobject, tmpmat), Transform(matrix_name, tmpmatname), Transform(bresinit, tmpbreso))
                self.wait(1)
            else: 
                self.play(Transform(matrix_mobject, tmpmat), Transform(matrix_name, tmpmatname), Transform(bresinit, tmpbreso), run_time=0.5)
                self.wait(0.25)
        self.wait(2)

        #Discard negative b1 values
        self.play(FadeOut(VGroup(matrix_mobject, matrix_name, tmpmat, tmpmatname)))
        discardtxt = Text("Discard iterations with\ncoefficients in the\nundesired direction").scale(0.7).to_edge(LEFT, buff=1)
        pbres = [bresfull[i] for i in [0,2,3,4]]
        posbres = Matrix(pbres).to_edge(RIGHT, buff=1).shift(DOWN*1).set_column_colors(WHITE, RED, BLUE, WHITE, WHITE)
        self.play(Create(discardtxt))
        self.wait(2)
        neg_row = tmpbreso.get_rows()[1]
        self.play(Indicate(neg_row))
        self.play(Transform(tmpbreso, posbres), FadeOut(bresinit))
        self.wait(2)
        self.play(FadeOut(discardtxt))
        wtcalctxt = Text("Calculate exposure weights\nas the weighted mean of\nbootstrap weights and\nt-statistics.").scale(0.7).to_edge(LEFT, buff=0.5)
        self.play(Create(wtcalctxt))
        self.wait(2)
        self.play(FadeOut(wtcalctxt))
        self.wait(0.5)

        #weighted mean
        wmeanform = MathTex(r"weighted~mean = \frac{\sum w_i*X_i}{\sum w_i}").scale(0.7).to_edge(LEFT, buff=0.5)
        e1wtcalc1 = MathTex(r"w_{E1}=\frac{0.6*2.3+0.8*2.8+0.9*3.1+0.8*2.8}{2.3+2.8+3.1+2.8}").scale(0.7).to_edge(LEFT, buff=0.5)
        e1wttitle = Text("E1 weight").scale(0.8).next_to(e1wtcalc1, UP, buff=0.5)
        e1wtcalc2 = MathTex(r"w_{E1}=\frac{8.65}{11}=2.8").scale(0.8).to_edge(LEFT, buff=1.5)
        # e1wtcalc3 = MathTex(r"w_{E1}=2.8", color=RED).to_edge(LEFT, buff=1.5)
        pbrgroup = VGroup(posbres, c1t, c2t, c3t, c4t, c5t)
        impcol1 = posbres.get_columns()[1]
        impcol2 = posbres.get_columns()[4]
        self.play(Indicate(impcol1), Indicate(impcol2))

        self.play(Create(wmeanform), pbrgroup.animate.scale(0.8).shift(RIGHT*0.5), FadeOut(VGroup(tmpbreso)))
        self.wait(1)
        self.play(FadeOut(wmeanform))
        self.play(Create(e1wtcalc1), Create(e1wttitle))
        self.wait(2)
        self.play(Transform(e1wtcalc1, e1wtcalc2))
        self.wait(1)
        self.play(FadeOut(pbrgroup))
        self.wait(0.5)
        self.play(FadeOut(VGroup(e1wtcalc2, e1wtcalc1, e1wttitle)))

        #Xq again
        matrix_mobject = matrix_mobject.to_corner(UP + LEFT, buff=1.5)
        matrix_name = MathTex("X_q").next_to(matrix_mobject, UP, buff=0.5)
        matgroup = VGroup(matrix_mobject, row1text, row2text, row3text, col1text, col2text, matrix_name)
        
        #w
        vector = [0.8, 0.2]
        vector_mobject = Matrix([[vector[0]], [vector[1]]]).set_row_colors(RED, BLUE).next_to(matrix_mobject, RIGHT, buff=2)
        vname = MathTex("w").next_to(vector_mobject, UP, buff=0.5)
        vectext1 = Text('E1', font_size=20, color=RED).next_to(vector_mobject, LEFT, buff=0.1).shift(UP*0.4)
        vectext2 = Text('E2', font_size=20, color=BLUE).next_to(vector_mobject, LEFT, buff=0.1).shift(DOWN*0.4)
        vecgroup = VGroup(vector_mobject, vectext1, vectext2, vname)
        dotsign = MathTex("\dot").next_to(matrix_mobject, RIGHT, buff=1)
        equals_sign = MathTex("=").next_to(vector_mobject, RIGHT, buff=1)

        #WQS
        wqsmat = [[1.6], [0.4], [1.0]]
        wqs_obj = Matrix(wqsmat).set_column_colors(PURPLE).next_to(equals_sign, RIGHT, buff=1)
        wqsname = MathTex("WQS").next_to(wqs_obj, UP, buff=0.5)
        row2text_wqs = Text('P2', font_size=20).next_to(wqs_obj, LEFT, buff=0.1)
        row1text_wqs = Text('P1', font_size=20).next_to(row2text_wqs, UP, buff=0.6)
        row3text_wqs = Text('P3', font_size=20).next_to(row2text_wqs, DOWN, buff=0.6)
        wqsgroup = VGroup(wqs_obj, wqsname, row1text_wqs, row2text_wqs, row3text_wqs)

        #Reshowing the Xq*w = WQS formula
        vector_mobject = vector_mobject.to_edge(LEFT, buff=2).shift(DOWN*2)
        vname = vname.next_to(vector_mobject, UP, buff=0.5)
        vectext1 = Text('E1', font_size=20, color=RED).next_to(vector_mobject, LEFT, buff=0.1).shift(UP*0.4)
        vectext2 = Text('E2', font_size=20, color=BLUE).next_to(vector_mobject, LEFT, buff=0.1).shift(DOWN*0.4)
        finalwttxt = Text("Final Weights").next_to(vector_mobject, UP, buff=1)
        self.play(Create(vector_mobject), Create(vname), Create(vectext1), Create(vectext2), Create(finalwttxt))
        self.wait(2)
        self.play(FadeOut(VGroup(vector_mobject, vname, vectext1, vectext2, finalwttxt)))
        # matrix_mobject = matrix_mobject.to_edge(LEFT, buff=1)
        vector_mobject = vector_mobject.next_to(matrix_mobject, RIGHT, buff=2)
        vname = vname.next_to(vector_mobject, UP, buff=0.5)
        vectext1 = Text('E1', font_size=20, color=RED).next_to(vector_mobject, LEFT, buff=0.1).shift(UP*0.4)
        vectext2 = Text('E2', font_size=20, color=BLUE).next_to(vector_mobject, LEFT, buff=0.1).shift(DOWN*0.4)
        allwqscalc = VGroup(matgroup, vecgroup, dotsign, equals_sign, wqsgroup) 
        self.play(Create(allwqscalc.shift(DOWN*2)))
        self.wait(2)

        #show regression
        self.play(FadeOut(trtext))
        step2text = Text("Stage 2: Regress the outcome on the\nWQS from Stage 1").scale(0.7).to_edge(UP)
        self.play(FadeOut(VGroup(matgroup, vecgroup, dotsign, equals_sign)), wqsgroup.animate.to_corner(DOWN + LEFT, buff=1.5), Create(step2text))
        self.wait(1)
        b1text = Tex(r"$\beta_1$").next_to(wqs_obj, RIGHT, buff=2)
        b0text = Tex(r"$\beta_0$").next_to(b1text, RIGHT, buff=2)
        astsign = MathTex("*").next_to(wqs_obj, RIGHT, buff=1)
        plussign = MathTex("+").next_to(b1text, RIGHT, buff=1)
        equals_sign = equals_sign.next_to(b0text, RIGHT, buff=1)
        ytext = Tex(r"$\hat{y}$").next_to(b0text, RIGHT, buff=2)
        glmgroup = VGroup(b1text, b0text, astsign, plussign, equals_sign, ytext)
        self.play(Create(glmgroup))
        self.wait(1.5)
        self.play(glmgroup.animate.shift(UP*1.5), wqsgroup.animate.shift(UP*1.5))
        regrpurpose = Tex(r"Coefficients of interest, particularly $\beta_1$, are\\derived from this 2nd stage regression.")
        self.play(Create(regrpurpose.to_edge(DOWN, buff=0.5)))
        self.wait(2)
        self.play(*[FadeOut(obj) for obj in self.mobjects]) #clear screen
        self.wait(0.5)

        #Summary
        header = Text("In Summary: There are two stages to WQS regression:").to_edge(UP, buff=1.5).scale(0.8)
        point1 = Tex(r"Stage 1: Training with bootstrapping and BFGS optimization\\to find final mixture weights $w$ and\\the weighted sum index $WQS$").scale(0.8).next_to(
            header, DOWN, buff=1).arrange(DOWN, center=False, aligned_edge=LEFT)  
        point2 = Tex(r"Stage 2: Applying the $WQS$ from Stage 1 in a linear regression\\to obtain final coefficients, especially the\\mixture coefficient $\beta_1$").scale(0.8).next_to(
            point1, DOWN, buff=1).arrange(DOWN, center=False, aligned_edge=LEFT)  
        self.play(Create(header))
        self.wait(0.5)
        self.play(Create(point1))
        self.wait(2)
        self.play(Create(point2))
        self.wait(4)
        

command1 = "manim WQSCalc.py WQSIntro -w -ql"
os.system(command1)

command2 = "manim WQSCalc.py WQSSteps -w -ql"
os.system(command2)