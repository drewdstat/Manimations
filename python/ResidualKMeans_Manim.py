# %%
import numpy as np
from manim import *
import scipy.linalg
from scipy import stats

def myols(x, y, qr = False, inclyhat = True):
    #normal XTY = XTXb
    #QR: QRTY = QRTQRb or RTY = RTRb

    #bhat ~ N(beta, (XTX)^-1*sigma2)
    #QR: bh ~ N(beta, (QRT*QR)^-1*sigma2) => bh ~ N(beta, RTR^-1*sigma2)
    
    if type(x) == list:
        x = np.array(x)
    if not np.all(x[:, 0] == 1):
        X = np.column_stack((np.ones(x.shape[0]), x))
    else: 
        X = x.copy()
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
    if inclyhat:
        out.update({"resids": resids})
        out.update({"preds": yhat})

    return out

# Manually create the datasets
disclustdata = np.array([
    [12.9072864549522, -13.8655875265834, -9.32509639647946, 9.29671880950218, -12.4103106180455, 13.1378980988871, 
     -15.5512611357576, 9.36990188381405, 4.70599048258087, 5.3575719215684, -7.84520423848616, 6.21740531616017, 
     4.42850695204969, -11.328832884077, -13.8606949071004, -12.092731630311, -10.7477231523501, 7.16364109681665, 
     2.00513047262845, 2.56113707997913, 8.88629206413317, 11.193881331324, -7.15536757999534, 13.5229463878833, 
     -10.5297697787056, -8.04030813892303, 4.88146693093143, 5.68091518965351, -11.4296599267599, -7.93635918002557, 
     5.92263799756632, -9.40209844420052, 12.2701820785113, -7.85500662615485, 6.45829006430443, -8.72790819343149, 
     6.29865667608811, -14.0088111396488, -8.56317992822009, 9.01788017212175, -15.4650009611778, -9.13328230469248, 
     10.8192993821622, 4.88645620831674, 3.66891918890787, -8.23033598015236, 11.8443697127635, -13.5693231882662, 
     -8.57896690919522, -9.36448171782152, 10.5919730666053, -14.8500593215551, 4.27765207439276, -14.7593856142357, 
     -12.3554567269773, -13.2808847382949, -11.4540432489808, 8.37804931111025, -9.24756555600529, -10.1251696716465, 
     13.1690811153322, 2.15931168005828, -10.2481907970829, 11.7749845349497, 7.39221411217916, -9.20276572784256, 
     8.12188289276387, 1.4773021710517, 4.28568189402121, -13.8070814812803, -13.7215510976276, 3.94633815452471, 
     14.4439548968978, 10.4979879028673, 4.80004056390762, -13.9160340569208, 3.58667727077382, -12.524622974859, 
     4.32424320117502, -6.56573362939798, -12.2944296684379, -10.5220587661165, 8.94561672455361, 5.30258882486666, 
     -9.62605098058026, -8.57141208293052, -9.70909059408363, -7.14235086604959, -13.0953593133157, 12.1991577997679, 
     10.7318175194179, 4.01175849595045, 2.7815911819778, 13.5001458370094, 12.9752691278232, -10.8601975301428, 
     -7.99027510970515, -8.8083979232158, 12.7713234384888, 10.5963651419625, 4.08382498008027, 6.66731666433312, 
     -7.0516498551167, -11.5559144806903, -8.00499744270056, 13.0064487441368, -7.91931742620949, -12.3567451463826, 
     11.5619068658111, 10.4545370305636, -11.2616140596238, 7.87993912103016, 11.7519322139906, 7.67807118443317, 
     5.74185446617995, 2.8350126963304, 3.71305292093974, -16.7803793972879, 12.8317157333851, -14.3271652867593, 
     -14.5150210900695, 9.6051661328747, -6.23292061494018, -9.74564525594075, -14.0441895513531, -13.7580757944319, 
     -11.4664443840918, 9.8646510467919, 9.17149699077796, 4.10705308661798, -7.10260066669179, -4.98460222364976, 
     4.05037209086719, 13.430167398569, 12.1081562322528, -10.7888409075685, -7.64184684310039, -7.13693095090488, 
     9.86270105364407, 9.81724564403809, 10.6663970892924, 11.8331698561089, -10.8498891970784, -9.41165488438699, 
     -7.27169146078699, -14.4539641973397, -14.5293548360522, 4.92089704902601, -11.1529466423417, 3.00600189193303, 
     3.25254862845697, 9.64481125201893, -7.00476475310136, 6.08884733612388, -9.49522717585057, -8.82090323585379, 
     -10.9267256250424, -12.0050401469331, 1.79549327392648, 7.90484199511813, 3.25464143903613, -10.8147519165158, 
     4.19712951192225, 10.437015395163, 11.7121828532089, -11.5234206773637, 7.71916259219174, 12.7227368653409, 
     12.2727165374659, 7.41344277837935, -8.85090544090087, -8.95273392649687, -7.44138255997816, 12.6342015754489, 
     -12.0397808171168, 4.70585153119701, 11.0590267109993, -9.60880204534265, 5.47614708142684, 8.43042705265563, 
     -11.7200910330229, 3.92687841279232, -9.86154471823209, 10.0292419197952, -9.31860381052492, 8.85105626019525, 
     -12.8282027876396, -11.8331102800479, 5.92528451746243, -11.8012400680886, -12.2818142940243, -8.55470413585607, 
     -6.96821520323018, -7.36777041293617, -6.77785599532573, 6.17620613496818, -9.71964540917223, -6.71010499996046, 
     -8.42049245029201, 4.31603835537551], 
    [3.05865664997986, -0.3872457168199, -3.69728929302774, 1.05349684918204, -7.63768906286119, 3.02017183546985, 
     -7.46865167746528, 3.53374049392218, -19.390974455819, -19.0671846875619, 2.45396470210839, -18.1076795182436, 
     -13.7756621947059, -5.26936149091573, -7.55276400542561, -1.9471270168737, -4.17243166944359, 2.38783462689348, 
     -18.0742163924145, -17.0332248202176, 3.39350440599737, 6.32248887323538, -2.81623165152935, 3.36527203377507, 
     -4.73876185232711, -1.74322699627706, -17.3065920679618, -18.0955146301344, 3.77749026246055, -3.7998794186356, 
     3.47847428480989, -1.97558822539435, 1.1667493870846, 1.19654323451946, 3.11985834506654, -2.19835874370071, 
     1.01281122176728, -1.46868023889813, 0.259842141014785, 1.31415116603205, -2.21978980475375, -0.0988420764787148, 
     3.27878888784973, -17.6992433420176, 5.87653747324755, 2.94184278886779, 1.95034262898199, -6.10532911186116, 
     -3.55983855131287, 0.292204924097449, 0.684606505574197, -7.64686075515991, -20.513841369584, -0.752843369114204, 
     -2.82005670051489, 2.16938963724798, -2.91329066778501, 2.32520613505046, -3.14939081092136, 0.0962893984178579, 
     1.36247951286886, -22.9271198121739, -1.33665704031756, 4.61469447316881, 1.68540670408116, -4.11311635472364, 
     2.02352112640016, -18.8155530242506, -17.9539735484816, -2.04433780364622, 1.6849269412814, -20.3989586124653, 
     -1.03337727866701, 2.47852772572186, -13.4126486034855, -5.16317103631438, -20.2780474686339, -8.89879902899566, 
     -18.7795847467515, -4.37542370478878, -3.59941566476815, -6.03184530647139, 0.528797264506819, -19.7373756846753, 
     -3.1159919365818, -1.41702693874996, -5.85654964903056, 0.206545951947806, -4.05980093026837, 2.2372163712095, 
     3.26417179982764, -21.7375041611854, -18.200566558411, 0.505295869230713, -0.530980148232643, -2.4380593000154, 
     2.45807531994816, -0.632231129738318, 2.67128475279075, 2.33998131928495, -17.60461933897, 4.5446696437575, 
     -1.04886527613451, -4.3583178778388, -4.00856457874881, -0.499104047426913, 0.0911350554303598, -3.34958688048436, 
     0.538398430607476, 5.64823281508422, -3.39121373407064, 6.36987399887915, 4.18628754040141, 3.1160883327649, 
     -16.5330765680779, -23.0340385234081, -15.1571166806068, -4.92710692301546, 4.28528263071289, -2.77930036804482, 
     -10.5641399979391, 1.75091541189388, -6.06736851530338, -3.05259244018714, -5.01175191362133, -8.0330493237686, 
     2.42868036424156, 4.19741908323486, 4.48830023731911, -21.6616636294585, 3.21424125250238, 3.07114101426995, 
     -20.2365757378016, 4.0205041816569, 1.36863836568815, -3.44194327478469, -3.08647353876523, -3.86079731091554, 
     3.36402443184257, 5.33129923618006, 1.38993971950214, 1.26422637103717, -3.88882775317899, -1.80557362267618, 
     -0.712285283789037, -3.36522938154752, -7.83174414962526, -17.1194656732154, -1.28108227953193, -19.4547897919807, 
     -20.1990600926321, 5.49672379722359, 1.92802659048411, -18.3165034868653, -4.39088232693637, -0.613536861644343, 
     -2.21198162678165, -4.14945355019623, -22.3639050828342, 4.64701022305763, -19.9978121293654, -3.65496801376204, 
     -15.7544277318209, 1.24033848550414, 3.83618956878889, 0.338259671472835, -15.1989730304084, 1.83062297466432, 
     1.45746090098496, 3.42079643816634, -0.737955353893708, -1.51328361072299, -1.88096010721194, -0.378606941577177, 
     -5.47299807746056, -17.7291358507334, 4.09310158402339, -0.628224548101546, -14.7417207582339, 1.06724233852949, 
     -4.13197217990468, -15.1624664595953, 1.00289798566108, 4.5591660022793, -0.194638164313298, 2.95331351687523, 
     -3.36183914957852, -1.68546621914506, -19.831147238757, -0.774094816674634, 1.12428517830481, -2.95639433682387, 
     0.474841232828246, -3.36278929562564, -3.20703303394737, -17.7092131894186, -3.6373148493348, -3.30544646713601, 
     -3.80360314016825, -18.9568576023095]
]).transpose()
disclusts = np.array(['D2', 'D1', 'D1', 'D2', 'D1', 'D2', 'D1', 'D2', 'D3', 'D3', 'D1', 'D3', 'D3', 'D1', 'D1', 'D1', 'D1', 'D2', 'D3', 
     'D3', 'D2', 'D2', 'D1', 'D2', 'D1', 'D1', 'D3', 'D3', 'D1', 'D1', 'D2', 'D1', 'D2', 'D1', 'D2', 'D1', 'D2', 'D1', 
     'D1', 'D2', 'D1', 'D1', 'D2', 'D3', 'D2', 'D1', 'D2', 'D1', 'D1', 'D1', 'D2', 'D1', 'D3', 'D1', 'D1', 'D1', 'D1', 
     'D2', 'D1', 'D1', 'D2', 'D3', 'D1', 'D2', 'D2', 'D1', 'D2', 'D3', 'D3', 'D1', 'D1', 'D3', 'D2', 'D2', 'D3', 'D1', 
     'D3', 'D1', 'D3', 'D1', 'D1', 'D1', 'D2', 'D3', 'D1', 'D1', 'D1', 'D1', 'D1', 'D2', 'D2', 'D3', 'D3', 'D2', 'D2', 
     'D1', 'D1', 'D1', 'D2', 'D2', 'D3', 'D2', 'D1', 'D1', 'D1', 'D2', 'D1', 'D1', 'D2', 'D2', 'D1', 'D2', 'D2', 'D2', 
     'D3', 'D3', 'D3', 'D1', 'D2', 'D1', 'D1', 'D2', 'D1', 'D1', 'D1', 'D1', 'D1', 'D2', 'D2', 'D3', 'D1', 'D1', 'D3', 
     'D2', 'D2', 'D1', 'D1', 'D1', 'D2', 'D2', 'D2', 'D2', 'D1', 'D1', 'D1', 'D1', 'D1', 'D3', 'D1', 'D3', 'D3', 'D2', 
     'D1', 'D3', 'D1', 'D1', 'D1', 'D1', 'D3', 'D2', 'D3', 'D1', 'D3', 'D2', 'D2', 'D1', 'D3', 'D2', 'D2', 'D2', 'D1',
     'D1', 'D1', 'D2', 'D1', 'D3', 'D2', 'D1', 'D3', 'D2', 'D1', 'D3', 'D1', 'D2', 'D1', 'D2', 'D1', 'D1', 'D3', 'D1', 
     'D1', 'D1', 'D1', 'D1', 'D1', 'D3', 'D1', 'D1', 'D1', 'D3'], dtype=str)

cohclustdata = np.array([
    [-7.93951687780519, 7.26809505003004, 5.55246304489878, 2.2032615819125, -4.0893232394224, -10.4066588161049, 
     -4.17486830951777, -8.87901709901523, 11.1534257989565, -7.91616095359262, -11.8885574169016, 6.18574871800703, 
     -8.20772763237549, -12.4969806155358, -9.54371478540399, 5.09387925433311, 8.06019208913303, 8.82816265319884, 
     -6.88444527631852, 5.69172209726896, -10.5232286726936, -8.479921596983, -6.78587923999256, -11.2473445320702, 
     7.98305172468233, 7.52412950218294, 6.21427981943531, -8.287505817078, -10.3838684168849, -8.53654887769384, 
     -9.73753906900856, 8.65536119862843, -7.05969912230398, -8.06309246781421, 7.25530267196541, -2.82548274650966, 
     4.86930240605769, -4.39061712742795, 7.78201443859012, 9.94500578471944, -8.51332088459803, 11.0830986845699, 
     3.41126902776938, -8.80558543421735, -8.4494347313917, -2.5581036893679, -8.13652303106804, -9.421947018936, 
     -5.05840142146506, -10.1728640466944, 5.76557866497971, -11.0433974228257, -6.13913491924022, -12.4819306974541, 
     7.45474780124148, -5.19765756462037, 4.75925835663574, 2.17827418077062, -8.38946346148624, -4.86644238112643, 
     -9.88835948557986, -4.85972758909019, 3.8123096602268, -11.8817731110445, 6.11010162969192, 5.85764129105185, 
     4.80705667873804, -6.53958615678367, 10.3025305331283, -7.73055632629948, -7.8762672491455, -5.49203741820293, 
     -7.31277388835302, 4.8943203635623, 4.39136259071349, -10.1809147083468, 3.08636429165546, 5.54330457966075, 
     4.35794365522708, -11.2681276338572, -4.70675504205743, -11.2205656087637, 7.62802466738289, -3.92148746556151, 
     -10.4945253644163, -9.12791303518184, -4.17697053243612, 6.83548687400104, 6.34159806434197, -6.21879792676031, 
     7.57043753880554, 7.69391964913422, -3.73012133181733, 4.3820287354364, -9.6576480496693, -10.0382807385959, 
     -6.68655604539896, 11.8052012431379, -6.11172758440039, -2.46053736990194, 6.71104247758964, -6.54854851613027, 
     -8.31791633510752, -2.4301406638962, -5.62527456664814, 11.5683803999104, -6.03014259753464, -6.85410318581012, 
     -6.88777900756693, -7.4897327461766, -12.2605611284382, 9.6986158174021, -8.44779944663339, -5.29664343837165, 
     8.74384749052719, -7.05422373295272, -4.86454048429038, -4.69762113011852, -4.661376155484, -9.52310264577564, 
     -6.55593658718466, -3.35357704159831, 3.87179507654715, -8.29819511602313, -9.57940407031572, 5.40237850018846, 
     -10.300703763611, -1.13330856030931, -13.224196052376, 8.18791565661965, 7.29884948935119, -6.56222063517549, 
     -7.54866179718073, -10.3262819031921, -5.07320677011064, -10.207778265392, -8.81767152365877, 5.57802646015661, 
     -7.00930937772669, -7.08192345065358, -3.10317398894038, -7.80654818842799, 13.0961250939508, -8.28773455603642, 
     7.2358542194554, -6.23570586946918, 6.26014804574874, -6.11994187101013, -6.51084202291854, -5.0317726575462,
    -13.4443641914551, -8.19004196741164, -7.53553592281911, 4.98706592811017, -9.41543790219744, -8.41138076524171, 
    8.03704717861583, 7.93547394192792, -8.36884855424464, -5.98197259281178, -5.45263640900615, -7.32519745422142, 
    5.8146052428812, 9.34569606030302, -5.60499742413109, 8.91472297972955, -7.58448256050951, -8.05657359688964, 
    -6.47328625839772, -2.85630103197713, -3.90391930018776, 9.20978625417424, -2.84830196268066, 10.8280484853391, 
    -6.72732767999903, 9.05462681706469, 8.10606117431224, -1.97812841756988, -6.29866840534776, 9.68788823715751, 
    -12.1278478571934, -2.44285606787953, 7.16144355559128, 5.62852084531405, 6.76181674183908, -3.83131953335459, 
    -6.36236400167226, 7.13805019620305, -3.67991891832606, 7.99545285058508, -12.486855824932, 7.29360277468265, 
    -4.08129602993787, -9.71926991133621, 0.0071616782096022, -11.5516748477835, 6.53211893140227, -3.43717789168928, 
    -10.7210303616756, -5.94836266831992], 
    [-15.97903712105, 12.9458186928232, 18.0234538882464, 22.5431456029903, -14.9852899243913, -16.0522349769307, 
     -16.4131566692802, -15.3041638148121, 14.5617048563526, -20.9637716589771, -15.5720636712687, 16.753258309099, 
     -16.6735400281551, -18.8404014472624, -12.6005485591131, 20.0096288921094, 13.3685729876261, 13.4797068788036, 
     -17.1651801547543, 13.0006051760464, -8.00337646123204, -17.9228478352414, -15.0683847819737, -17.7941503878206, 
     15.2407797242301, 13.3650354365001, 15.8875390020761, -17.7108020378004, -12.5155460133537, -12.8615455160829, 
     -18.2925514646256, 13.8966170420202, -12.7916453713742, -13.791568042845, 20.0340922515572, -14.1580869937024, 
     17.5589439315788, -13.7376008421583, 14.384289594844, 16.2750429937428, -15.1358934320773, 14.0096258673949, 
     15.6863867850638, -17.626759429893, -12.0232864078002, -12.6460884231111, -18.1730827277493, -13.4797693693792, 
     -15.9364562024629, -17.3858726271388, 16.5929319002172, -18.3079957473086, -17.7190404621744, -14.106053686699, 
     12.5459756614675, -15.4807507413352, 16.6340330637973, 17.2759083206166, -16.501192974652, -14.374097455104, 
     -14.5749683151281, -19.0958637905045, 16.6857229823666, -15.6254195735419, 11.4524130218892, 14.87514472424, 
     15.8583954592836, -15.0956923508774, 8.24829781221956, -13.6646549159083, -12.551563074467, -16.126915415409, 
     -17.1737450899312, 16.2441749737713, 19.2182575947523, -13.0478880892386, 19.2616659952229, 16.747767319734, 
     16.3186670003662, -13.7918334570449, -17.1905780936159, -12.5243237541035, 14.9665584242601, -16.8727977638975, 
     -13.8167755539813, -15.7149828319867, -15.8646970225234, 15.1201889772794, 14.9751857609049, -17.113907473771, 
     11.152331670308, 11.0287204794045, -17.8700586338987, 17.0023652414937, -19.7407723407635, -17.3535968169337, 
     -17.087499594284, 11.0570896727753, -12.8201980989615, -17.2748500860335, 13.4220927260685, -13.1846312164861, 
     -16.130985448086, -12.4640793251768, -15.033515511934, 9.87223486033879, -11.4825799918433, -8.61329077242763, 
     -16.5709079560253, -16.5290930532736, -16.8583138188773, 15.7102163997674, -17.3773513889846, -15.0565601149469, 
     13.2719862467604, -14.5971101884719, -14.9941786732187, -14.5010098769183, -17.8281147717981, -17.8113798625032, 
     -17.8635655535668, -12.0834791126924, 16.3198233970719, -14.6919076568772, -11.5047341074086, 13.7516993693836, 
     -14.9898078680718, -15.9163296435747, -17.3878598114927, 14.067658636921, 17.7024705031024, -18.0353026571363, 
     -15.8009873578859, -12.3861149795802, -12.2258233836108, -14.3259534333443, -15.7774987241783, 15.7948003667708, 
     -16.5445224484495, -19.6697010640364, -13.5559915314757, -15.3302273250956, 9.89901828949704, -18.6318871150969, 
     15.8623630344459, -13.2018572252749, 14.2285343167573, -15.1668195405119, -12.9607289171173, -15.7500403557412, 
     -13.5418351447128, -11.0954146303349, -15.6984910343664, 14.6524719707391, -16.4868459532969, -13.1923464510662, 
     15.7880001963728, 11.1064850510405, -19.5308700443042, -17.1076683055285, -9.21697215150987, -13.732110620967, 
     14.8605649639236, 12.4705169368856, -11.8472942235051, 15.0027515662154, -17.2552700635019, -15.02887976862, 
     -16.462467041772, -14.162748171109, -17.7152965529668, 13.5117868600885, -16.7082047857344, 11.3163740423505, 
     -15.411664949391, 12.5722373625631, 18.6325439436059, -11.9182850414649, -14.0104183266979, 9.96003061419084, 
     -17.1565494559726, -16.0938767902334, 17.2918546851151, 16.2727388132251, 14.2535742475261, -16.6752367586904, 
     -18.9146794966788, 18.833809893321, -14.4559885601116, 15.3561990328183, -13.8778831088715, 14.7731999675186, 
     -17.3301741200995, -15.041379313379, -16.023865049735, -12.9183874055035, 15.2191284088183, -15.2305556190821, 
     -18.0161007919109, -16.211209298488]
]).transpose()

cohclusts = np.array(['C1', 'C2', 'C2', 'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C1', 'C2', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C1', 
     'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C1', 'C2', 'C1', 'C2', 'C1', 
     'C2', 'C2', 'C1', 'C2', 'C2', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C1', 'C1', 'C2', 'C1', 'C2', 
     'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C2', 'C2', 'C2', 'C1', 'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C1', 
     'C2', 'C2', 'C2', 'C1', 'C1', 'C1', 'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C1', 'C2', 'C2', 'C1', 'C2', 'C1', 
     'C1', 'C1', 'C2', 'C1', 'C1', 'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C1', 
     'C2', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C1', 'C2', 'C1', 'C1', 'C1', 'C2', 'C2', 'C1', 'C1', 
     'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C2', 'C1', 'C2', 'C1', 'C1', 'C1', 'C1', 'C1', 
     'C1', 'C2', 'C1', 'C1', 'C2', 'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C1', 'C2', 'C1', 'C1', 'C1', 'C1', 'C1', 
     'C2', 'C1', 'C2', 'C1', 'C2', 'C2', 'C1', 'C1', 'C2', 'C1', 'C1', 'C2', 'C2', 'C2', 'C1', 'C1', 'C2', 'C1', 'C2', 
     'C1', 'C2', 'C1', 'C1', 'C1', 'C1', 'C2', 'C1', 'C1', 'C1'], dtype=str)

naiveclusts = np.array(['N2', 'N3', 'N3', 'N5', 'N1', 'N2', 'N1', 'N2', 'N2', 'N4', 'N1', 'N2', 'N4', 'N1', 'N1', 'N3', 'N3', 'N5', 'N4', 
                        'N2', 'N2', 'N2', 'N1', 'N2', 'N3', 'N3', 'N2', 'N4', 'N1', 'N1', 'N2', 'N3', 'N2', 'N1', 'N5', 'N1', 'N5', 'N1', 
                        'N3', 'N5', 'N1', 'N3', 'N5', 'N4', 'N2', 'N1', 'N2', 'N1', 'N1', 'N1', 'N5', 'N1', 'N4', 'N1', 'N3', 'N1', 'N3', 
                        'N5', 'N1', 'N1', 'N2', 'N4', 'N3', 'N2', 'N5', 'N3', 'N5', 'N4', 'N2', 'N1', 'N1', 'N4', 'N2', 'N5', 'N5', 'N1', 
                        'N2', 'N3', 'N2', 'N1', 'N1', 'N1', 'N5', 'N4', 'N1', 'N1', 'N1', 'N3', 'N3', 'N2', 'N5', 'N2', 'N4', 'N5', 'N2', 
                        'N1', 'N1', 'N3', 'N2', 'N2', 'N2', 'N2', 'N1', 'N1', 'N1', 'N5', 'N1', 'N1', 'N2', 'N2', 'N1', 'N5', 'N2', 'N2', 
                        'N2', 'N4', 'N4', 'N1', 'N2', 'N1', 'N1', 'N2', 'N3', 'N1', 'N1', 'N3', 'N1', 'N2', 'N2', 'N2', 'N3', 'N1', 'N4', 
                        'N2', 'N2', 'N1', 'N1', 'N3', 'N2', 'N2', 'N2', 'N2', 'N3', 'N1', 'N3', 'N1', 'N3', 'N4', 'N1', 'N4', 'N4', 'N2', 
                        'N1', 'N2', 'N1', 'N1', 'N3', 'N3', 'N4', 'N2', 'N4', 'N1', 'N2', 'N5', 'N2', 'N3', 'N4', 'N2', 'N2', 'N2', 'N1', 
                        'N3', 'N1', 'N5', 'N1', 'N2', 'N5', 'N1', 'N4', 'N5', 'N1', 'N4', 'N3', 'N5', 'N3', 'N2', 'N1', 'N3', 'N4', 'N3', 
                        'N1', 'N3', 'N1', 'N1', 'N1', 'N4', 'N3', 'N1', 'N1', 'N4'], dtype=str)

# %%

#Combine datasets
combclustdata = (disclustdata + cohclustdata)/2
combclusts = cohclusts + '.' + disclusts

def one_hot_encode_numpy(labels, as_ols_predmatrix = True):
    unique_labels = np.unique(labels)
    one_hot_matrix = np.zeros((len(labels), len(unique_labels)))
    for i, label in enumerate(labels):
        one_hot_matrix[i, np.where(unique_labels == label)[0][0]] = 1
    if as_ols_predmatrix:
        one_hot_matrix = np.column_stack((np.ones(len(labels)), one_hot_matrix[:,1:]))
    return one_hot_matrix

#Get residuals from combined dataset
def residualize_bylabel(labels, feature_matrix):
    pred_matrix = one_hot_encode_numpy(labels, True)
    resid_matrix = []
    for column in feature_matrix.T:
        resids_temp = myols(pred_matrix, column)['resids']
        resid_matrix.append(resids_temp)
    return np.array(resid_matrix).T

residclustdata = residualize_bylabel(cohclusts, combclustdata)

#Scale datasets
def scale_np(arr):
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    return (arr - means) / stds

combclustdata_scaled = scale_np(combclustdata)
residclustdata_scaled = scale_np(residclustdata)
cohclustdata_scaled = scale_np(cohclustdata)
disclustdata_scaled = scale_np(disclustdata)

# Calculate centers without pandas
def calculate_cluster_centers_numpy(data, clusters):
    features = data # All columns must be floats
    unique_clusters = np.sort(np.unique(clusters)) #order cluster names 
    cluster_centers = []
    for cluster in unique_clusters:
        cluster_points = features[clusters == cluster]
        cluster_centers.append(np.mean(cluster_points, axis=0))
    return np.array(cluster_centers)

disclust_centers = calculate_cluster_centers_numpy(disclustdata_scaled, disclusts)
cohclust_centers = calculate_cluster_centers_numpy(cohclustdata_scaled, cohclusts)
combclust_centers = calculate_cluster_centers_numpy(combclustdata_scaled, combclusts)
naive_centers = calculate_cluster_centers_numpy(combclustdata_scaled, naiveclusts)
residclust_centers = calculate_cluster_centers_numpy(residclustdata_scaled, disclusts)

#Manim prompt for the scene:
# Write a manim scene in which there is a 2-D axis that is first populated with a scatter plot of small, translucent points from the numpy ndarray disclustdata_scaled 
# and large points from the numpy ndarray disclust_centers, each colored by the numpy string array disclusts. Wait 4 seconds, after which add to that scatter plot small, 
# translucent points from the numpy ndarray cohclustdata_scaled and large points from the numpy ndarray cohclust_centers, each colored by the numpy string array cohclusts. 
# Wait 4 seconds, after which replace all points in the scatter plot using a Transform transition with small, translucent points from the numpy ndarray combclustdata_scaled 
# and large points from the numpy ndarray combclust_centers, all colored by the numpy string array combclusts. Wait 4 seconds, and then replace the large points with the 
# numpy ndarray naive_centers and change the colors to be colored by the numpy string array naiveclusts. Wait 4 seconds, and use a Transform transition to replace all the 
# points in the scatterplot with a scatterplot of small, translucent triangle-shaped points from the numpy ndarray disclustdata_scaled and large, translucent triangle-shaped 
# points from the numpy ndarray disclust_centers, as well as adding small square-shaped points from the numpy ndarray residclustdata_scaled and large square-shaped points from 
# the numpy ndarray residclust_centers. These points should all be colored by the numpy string array disclusts. Wait 10 seconds.
# %%
class ClusterVisualization2D(Scene):
    def construct(self):
        #Title Intro
        title = Text('Visualizing Residual K-Means:\nRemoving Nuisance Clustering').scale(0.8)
        self.play(Create(title))
        self.wait(2)
        self.play(FadeOut(title))
        
        # Create 2D axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"include_numbers": True},
        ).scale(0.9).to_edge(UP, buff=0.5)
        self.add(axes)

        # Helper function to create scatter points
        def create_scatter_points(data, labels, colors, shape=Dot, size=0.6, opacity=0.6):
            points = []
            for (x, y), label in zip(data[:, :2], labels):
                color = colors[label]
                points.append(shape(color=color, fill_opacity=opacity).scale(size).move_to(axes.c2p(x, y)))
            return points

        # Define colors for clusters
        unique_disclusts = np.unique(disclusts)
        disclust_colors = {label: color for label, color in zip(unique_disclusts, [BLUE, GREEN, RED])}

        unique_cohclusts = np.unique(cohclusts)
        cohclust_colors = {label: color for label, color in zip(unique_cohclusts, [YELLOW, ORANGE])}

        unique_combclusts = np.unique(combclusts)
        combclust_colors = {label: color for label, color in zip(unique_combclusts, [PURPLE, TEAL, GOLD, DARK_BLUE, GREEN_E, LIGHT_PINK])}

        unique_naiveclusts = np.unique(naiveclusts)
        naiveclust_colors = {label: color for label, color in zip(unique_naiveclusts, [MAROON, PINK, GRAY, WHITE, DARK_BROWN])}

        final_disclust_colors = {label: color for label, color in zip(unique_disclusts, [BLUE_A, GREEN_A, RED_A])}

        # Step 1: Plot disclustdata_scaled and disclust_centers

        step1txt = Text("Imagine we have two features clustered by 3\nlatent disease states we want to discover.").scale(0.6).to_edge(DOWN, buff=0.5)
        axlab1 = Text("Feature 1").scale(0.4).next_to(axes, LEFT, buff=-1).shift(DOWN*0.6)
        axlab2 = Text("Feature 2").scale(0.4).next_to(axes, UP, buff=-0.25).shift(LEFT*0.8)
        
        center_size = 2
        disclust_points = create_scatter_points(disclustdata_scaled, disclusts, disclust_colors)
        disclust_centers_points = create_scatter_points(disclust_centers, unique_disclusts, disclust_colors, size=center_size, opacity=1.0)
        self.play(Create(step1txt))
        self.wait(0.5)
        self.play(Create(axlab1), Create(axlab2))
        self.play(AnimationGroup(*[FadeIn(point) for point in disclust_points + disclust_centers_points]))
        self.wait(4)
        self.play(FadeOut(step1txt))
        self.wait(1)

        # Step 2: Add cohclustdata_scaled and cohclust_centers
        step2txta = Text("To better characterize the broader population, in this example\nwe collate data across two observational cohorts.").scale(0.6).to_edge(DOWN, buff=0.5)
        step2txtb = Text("However, these cohorts induce clustering (orange & yellow) due to\ntemporal, spatial, and methodological variation.").scale(0.6).to_edge(DOWN, buff=0.5)
        self.play(Create(step2txta))
        self.wait(4)
        self.play(FadeOut(step2txta), FadeIn(step2txtb))
        cohclust_points = create_scatter_points(cohclustdata_scaled, cohclusts, cohclust_colors)
        cohclust_centers_points = create_scatter_points(cohclust_centers, unique_cohclusts, cohclust_colors, size=center_size, opacity=1.0)
        self.play(AnimationGroup(*[FadeIn(point) for point in cohclust_points + cohclust_centers_points]))
        self.wait(4)
        self.play(FadeOut(step2txtb))

        # Step 3: Replace all points with combclustdata_scaled and combclust_centers
        step3txt = Text("This nuisance clustering by cohort creates the illusion\nof 6 clusters for each combination of cohort and\ndisease state, masking the clusters of interest.").scale(0.6).to_edge(DOWN, buff=0.5)
        self.play(Create(step3txt))
        combclust_points = create_scatter_points(combclustdata_scaled, combclusts, combclust_colors)
        combclust_centers_points = create_scatter_points(combclust_centers, unique_combclusts, combclust_colors, size=center_size, opacity=1.0)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    *[Transform(old, new) for old, new in zip(disclust_centers_points + cohclust_centers_points, combclust_centers_points)] #[0:(len(combclust_centers_points)-2)]
                ),
                AnimationGroup(*[FadeOut(point) for point in cohclust_points + disclust_points]), 
                AnimationGroup(*[FadeIn(point) for point in combclust_points]), 
                FadeIn(combclust_centers_points[len(combclust_centers_points)-1])
            )
        )
        self.wait(5)
        self.play(FadeOut(step3txt))

        # Step 4: Replace large points with naive_centers
        step4txt = Text("In this case, naively applying k-means clustering to\nthese data results in 5 distinct clusters.").scale(0.6).to_edge(DOWN, buff=0.5)
        naiveclust_points = create_scatter_points(combclustdata_scaled, naiveclusts, naiveclust_colors)
        naive_centers_points = create_scatter_points(naive_centers, unique_naiveclusts, naiveclust_colors, size=center_size, opacity=1.0)
        self.play(FadeIn(step4txt))
        self.play(
            AnimationGroup(
                AnimationGroup(*[FadeIn(point) for point in naive_centers_points]),
                AnimationGroup(*[Transform(old, new) for old, new in zip(combclust_points, naiveclust_points)]),
                AnimationGroup(*[FadeOut(point) for point in combclust_centers_points + disclust_centers_points + cohclust_centers_points])
            )
        )
        self.wait(4)
        self.play(FadeOut(step4txt))

        # Step 5: Replace all points with disclustdata_scaled (triangles) and residclustdata_scaled (squares)
        step5txta = Text("Residual k-means involves regressing each feature on cohort\nand k-means clustering the scaled residuals.").scale(0.6).to_edge(DOWN, buff=0.5)
        step5txtb = Text("Here are the original scaled disease state clusters (dots).").scale(0.6).to_edge(DOWN, buff=0.5)
        step5txtc = Text("And here are the scaled residuals (triangles), overlapping the\noriginal disease state clustering structure.").scale(0.6).to_edge(DOWN, buff=0.5)
        step5txtd = Text("These are the new cluster centers after residual k-means (triangles),\nclosely matching the original centers.").scale(0.6).to_edge(DOWN, buff=0.5)
        self.play(AnimationGroup(*[FadeOut(point) for point in naive_centers_points + naiveclust_points + combclust_points]))
        self.play(Create(step5txta))
        self.wait(4)
        self.play(FadeOut(step5txta), FadeIn(step5txtb))
        disclust_translucent_points = create_scatter_points(disclustdata_scaled, disclusts, disclust_colors)
        disclust_translucent_centers = create_scatter_points(disclust_centers, unique_disclusts, disclust_colors, size=2.5, opacity=0.9)
        residclust_triangle_points = create_scatter_points(residclustdata_scaled, disclusts, final_disclust_colors, shape=Triangle, size=0.02, opacity=0.3)
        residclust_triangle_centers = create_scatter_points(residclust_centers, unique_disclusts, final_disclust_colors, shape=Triangle, size=0.3, opacity=0.7)

        self.play(
            AnimationGroup(*[FadeIn(point) for point in disclust_translucent_points + disclust_translucent_centers])
        )
        self.wait(4)
        self.play(FadeOut(step5txtb), FadeIn(step5txtc))
        self.play(AnimationGroup(*[FadeIn(point) for point in residclust_triangle_points]))
        self.wait(4)
        self.play(FadeOut(step5txtc), FadeIn(step5txtd))
        self.play(AnimationGroup(*[FadeIn(point) for point in residclust_triangle_centers]))
        self.play(AnimationGroup(*[FadeOut(point) for point in residclust_triangle_points + disclust_translucent_points]))
        self.wait(10)
