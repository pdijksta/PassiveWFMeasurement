#(Identifier, streaker betax, streaker alphax, screen betax, R12, R11, phase advance, K1Ls
#('Optics 1', 10, 0, 100, 15, 0, 90, [-0.5, 1.23, 4.09,....])
#print('(Identifier, streaker betax, streaker alphax, screen betax, R12, R11, phase advance, K1Ls)')



### ARAMIS


quadNames=['SARUN15.MQUA080','SARUN16.MQUA080','SARUN17.MQUA080','SARUN18.MQUA080','SARUN19.MQUA080','SARUN20.MQUA080','SARBD01.MQUA020']

k1=[-2.675689435128679, 2.507107341321333, 1.534902208915763,1.389193277247406,-2.839259815798751,2.057266014806838,-0.5006955769293514]
k1a=[-2.849999999974721,2.560985202693461,1.391095627869793,1.336403614777685,-2.417256186871410,1.555253158719475,-0.04304492284097716]
k2=[-2.839190857780153, 2.496631435961427, 0.1964714303307858,1.355788652120069,-2.578141113979930,1.754252917520067,-0.2190583437308177]
k3=[-2.835049440970081, 2.017229324363440, 0.01979149831968566,1.355788652120069,-2.578141113979930,1.754252917520067,-0.2190583437308177]
k4=[-2.722096943074689, 1.513077849646517, 0.3093367069824747,1.355788652120069,-2.578141113979930,1.754252917520067,-0.2190583437308177]
k5=[-2.815796600917987, 1.172809302247960, 0.4829149631005829,1.355788652120069,-2.578141113979930,1.754252917520067,-0.2190583437308177]
k6=[-2.832141571171347, 0.8358291111665472, 0.6560775719265421,1.355788652120069,-2.578141113979930,1.754252917520067,-0.2190583437308177]
k7=[-2.843572546273418, 0.5293345918797998, 0.7974849079408017,1.355788652120069,-2.578141113979930,1.754252917520067,-0.2190583437308177]

L1=0.08
kL1 = [L1*x for x in k1]
kL1a = [L1*x for x in k1a]
kL2 = [L1*x for x in k2]
kL3 = [L1*x for x in k3]
kL4 = [L1*x for x in k4]
kL5 = [L1*x for x in k5]
kL6 = [L1*x for x in k6]
kL7 = [L1*x for x in k7]

#alpha_x=0 @ streaker location
ar1=['Optics 1', 2.5, -0.0008981, 89.99, 15, 0, 90,kL1] #max beta_y=300 m -> 100 um for 500 nm emittance
ar1a=['Optics 1a', 2.5, 0.0020903, 89.86, 15, 0, 90,kL1a] #larger beta_y at the screen, smaller max beta_y=250 m -> 100 um for 500 nm emittance
ar2=['Optics 2', 5, -0.0006851, 44.99, 15, 0, 90,kL2] # max beta_y=100-120 m for all optics from now on
ar3=['Optics 3', 10, 0.0007783, 22.5, 15, 0, 90,kL3]
ar4=['Optics 4', 15, 0.0004145, 15.0, 15, 0, 90,kL4]
ar5=['Optics 5', 20, -0.0006088, 11.25, 15, 0, 90,kL5]
ar6=['Optics 6', 25, -0.0007742, 9.0, 15, 0, 90,kL6]
ar7=['Optics 7', 30, -0.0009361, 7.5, 15, 0, 90,kL7]

#alpha_x variable at the streaker
fix=[-0.8215394696441847,-2.627957708897591,1.948805717574930,0.08324019299676864] #R11=2; R12=15 fixed for all cases

k8=[-2.849999999986756,2.229926103527097,1.795090018830287]+fix
k9=[-2.839740635257146,1.896252405510051,1.410579034526927]+fix
k10=[-2.446029810250162,0.8743227171583621,1.883394248030741]+fix
k11=[-1.160387525272524,-0.8811652066013742,2.546057721518306]+fix
k12=[-2.849999999703262,-0.07663459738290729,2.216294525812129]+fix
k13=[-2.849999999983728,-0.5684030998315928,2.367907655445079]+fix
k14=[-2.515864047753215,-1.358805776916345,2.563882759003665]+fix

kL8 = [L1*x for x in k8]
kL9 = [L1*x for x in k9]
kL10 = [L1*x for x in k10]
kL11 = [L1*x for x in k11]
kL12 = [L1*x for x in k12]
kL13 = [L1*x for x in k13]
kL14 = [L1*x for x in k14]

ar8=['Optics 8', 2.5,0.3315, 89.89, 15, 2, 90,kL8]
ar9=['Optics 9', 5.0,0.6657, 45.06, 15, 2, 90,kL9]
ar10=['Optics 10', 10.0,1.3321, 22.53, 15, 2, 90,kL10]
ar11=['Optics 11', 15.0,1.999,15.02, 15, 2, 90,kL11]
ar12=['Optics 12', 20.0,2.668,11.26, 15, 2, 90,kL12]
ar13=['Optics 13', 25.0,3.335,9.01, 15, 2, 90,kL13]
ar14=['Optics 14', 30.0,3.999,7.51, 15, 2, 90,kL14]

quadNames.append('SARBD02.MQUA030')
aramis_quads = quadNames
aramis_optics = [
        ar1,
        ar1a,
        ar2,
        ar3,
        ar4,
        ar5,
        ar6,
        ar7,
        ar8,
        ar9,
        ar10,
        ar11,
        ar12,
        ar13,
        ar14,
        ]

for a in aramis_optics:
    a[-1].append(0.)
    a[0] = a[0].replace('Optics', 'SARUN18-UDCP020')


### ATHOS POST-UNDULATOR

#Identifier, streaker betax, streaker alphax, screen betax, R12, R11, phase advance, K1Ls

quadNames=['SATUN22.MQUA080','SATMA02.MQUA010','SATMA02.MQUA020','SATMA02.MQUA040','SATMA02.MQUA050','SATMA02.MQUA070','SATBD01.MQUA010','SATBD01.MQUA030','SATBD01.MQUA050','SATBD01.MQUA070','SATBD01.MQUA090']

QM1=[4.178613916247179,-3.105571097756030,5.599986006176609,-2.868134034737537] # @streaker, beta_x=1.01012 m, @screen beta_x=220m 5.6m^-2 limit , better in y
QM1B=[1.011035788371912,1.478742377078101,4.279999993501157,-3.873096541807198] # @streaker, beta_x=1.01012 m, @screen beta_x=219m 4.28m^-2 limit, !back up, signs are correct! +++-

QM2=[5.246202884528229,-3.958270299291292,4.841574908760317,-2.915051871844490] # @streaker, beta_x=2.50881 m, @screen beta_x=88.5m 5.25m^-2 limit
QM2B=[4.279998716820764,-1.804821158078722,4.279999994968661,-3.134981443452437] # @streaker, beta_x=2.51078 m, @screen beta_x=88.5m 4.28m^-2 limit, backup

QM3=[4.878485674961142,-3.559646916851266,4.153733961804121,-2.753887141746631] # @streaker, beta_x=5.00939 m, @screen beta_x=44.3m 4.9m^-2 limit
QM3B=[4.199999995280455,-2.569402055950024,4.199999970709370,-3.424039731930264] # @streaker, beta_x=5.01046 m, @screen beta_x=44.3m 4.28m^-2 limit, backup

QM4=[3.553876009541617,-3.055820617590352,4.220150297193038,-3.015944985300997] # @streaker, beta_x=9.99006 m, @screen beta_x=22.2m 4.28m^-2 limit

QM5=[2.501590370488222,-2.685658518431310,4.279999983400799,-2.983814080614478] # @streaker, beta_x=14.9894 m, @screen beta_x=14.8m 4.28m^-2 limit

QM6=[2.774180668006945,-3.680779029072487,4.194464794292636,-2.284100109431831] # @streaker, beta_x=19.9872 m, @screen beta_x=11.1m 4.28m^-2 limit

QM7=[3.042334552493248,-4.199999996750876,3.820497766101923,-1.470299214300066] # @streaker, beta_x=24.9869  m, @screen beta_x=8.9m 4.28m^-2 limit

QM8=[3.050934859398875,-4.199999981837917,3.392281865967324,-0.8587432449134311] # @streaker, beta_x=29.9871   m, @screen beta_x=7.4m 4.28m^-2 limit

#Transport Quads K1 values:
QTMA15=[1.103585863863388,-2.627303531903210]
QTBD15=[1.087834034343101,-2.506256739449366,1.693228000031815,-2.557723926080568,0.7984360365715055]

#for QM1-5
QTMA20A=[-0.01834499922465258,-0.4254829576843816]
QTBD20A=[1.113351264646486,-2.197972116354360,1.225692505080959,-3.939426300961390,1.241743467411070]

#for QM6
QTMA20B=[0.6204373026911869,-1.946635645930276]
QTBD20B=[1.272655140340403,-2.211985331507563,1.074129065788922,-3.877923018008127,1.237912224639985]

#for QM7
QTMA20C=[0.7993004991546979,-2.347066715384223]
QTBD20C=[1.363732375309183,-2.559923683483548,0.9406342658957449,-3.100697470352157,1.157365925032256]

#for QM8
QTMA20D=[1.102973328903388,-2.700600242902217]
QTBD20D=[1.241877721216371,-2.195667795345614,1.263151289866165,-3.912249570139790,1.209308988993182]

#for QM1-4
QTMA10A=[-1.148429484933967,-0.7937775930634683]
QTBD10A=[1.181177690534110,-2.457290126850203,1.613801828961291,-2.479983316270420,0.7308980690147209]

#for QM5
QTMA10B=[-2.073199607742802,0.8372193544537587]
QTBD10B=[0.9641356560432376,-2.397709649042056,1.693140511291732,-2.423980358262178,0.7263773330751191]

#for QM6
QTMA10C=[-2.583883686536702,1.657798428197393]
QTBD10C=[0.7596804378366938,-2.278311719891428,1.750755129712383,-2.449808811213372,0.7255270450556310]

L1=0.08
L2=0.3

QM1N= [x * L1 for x in QM1]
QM1BN= [x * L1 for x in QM1B]

QM2N= [x * L1 for x in QM2]
QM2BN= [x * L1 for x in QM2B]

QM3N= [x * L1 for x in QM3]
QM3BN= [x * L1 for x in QM3B]

QM4N= [x * L1 for x in QM4]
QM5N= [x * L1 for x in QM5]
QM6N= [x * L1 for x in QM6]
QM7N= [x * L1 for x in QM7]
QM8N= [x * L1 for x in QM8]

QTMA15N = [x * L1 for x in QTMA15]
QTBD15N = [x * L2 for x in QTBD15]

QTMA20AN = [x * L1 for x in QTMA20A]
QTBD20AN = [x * L2 for x in QTBD20A]

QTMA20BN = [x * L1 for x in QTMA20B]
QTBD20BN = [x * L2 for x in QTBD20B]

QTMA20CN = [x * L1 for x in QTMA20C]
QTBD20CN = [x * L2 for x in QTBD20C]

QTMA20DN = [x * L1 for x in QTMA20D]
QTBD20DN = [x * L2 for x in QTBD20D]

QTMA10AN = [x * L1 for x in QTMA10A]
QTBD10AN = [x * L2 for x in QTBD10A]

QTMA10BN = [x * L1 for x in QTMA10B]
QTBD10BN = [x * L2 for x in QTBD10B]

QTMA10CN = [x * L1 for x in QTMA10C]
QTBD10CN = [x * L2 for x in QTBD10C]

#B for back up no R-> R12=15 m, R10->R12=10m,R20->R12=20m
#Optics 1-5 should work 3-5 fs resolution

#@streaker, beta_x=1.01012 m
opt1=QM1N+QTMA15N+QTBD15N
opt1B=QM1BN+QTMA15N+QTBD15N
opt1R10=QM1N+QTMA10AN+QTBD10AN
opt1BR10=QM1BN+QTMA10AN+QTBD10AN
opt1R20=QM1N+QTMA20AN+QTBD20AN
opt1BR20=QM1BN+QTMA20AN+QTBD20AN

# @streaker, beta_x=2.50881 m,
opt2=QM2N+QTMA15N+QTBD15N
opt2B=QM2BN+QTMA15N+QTBD15N
opt2R10=QM2N+QTMA10AN+QTBD10AN
opt2BR10=QM2BN+QTMA10AN+QTBD10AN
opt2R20=QM2N+QTMA20AN+QTBD20AN
opt2BR20=QM2BN+QTMA20AN+QTBD20AN

# @streaker, beta_x=5.00939 m,
opt3=QM3N+QTMA15N+QTBD15N
opt3B=QM3BN+QTMA15N+QTBD15N
opt3R10=QM3N+QTMA10AN+QTBD10AN
opt3BR10=QM3BN+QTMA10AN+QTBD10AN
opt3R20=QM3N+QTMA20AN+QTBD20AN
opt3BR20=QM3BN+QTMA20AN+QTBD20AN

# @streaker, beta_x=9.99006 m,
opt4=QM4N+QTMA15N+QTBD15N
opt4R10=QM4N+QTMA10AN+QTBD10AN
opt4R20=QM4N+QTMA20AN+QTBD20AN

# @streaker, beta_x=14.9894 m,
opt5=QM5N+QTMA15N+QTBD15N
opt5R10=QM5N+QTMA10BN+QTBD10BN
opt5R20=QM5N+QTMA20AN+QTBD20AN

# @streaker, beta_x=19.9872 m
opt6=QM6N+QTMA15N+QTBD15N
opt6R10=QM6N+QTMA10CN+QTBD10CN
opt6R20=QM6N+QTMA20BN+QTBD20BN

# @streaker, beta_x=24.9869  m
opt7=QM7N+QTMA15N+QTBD15N
opt7R20=QM7N+QTMA20CN+QTBD20CN

# @streaker, beta_x=29.9871   m
opt8=QM8N+QTMA15N+QTBD15N
opt8R20=QM8N+QTMA20DN+QTBD20DN

#alpha_x=0 @ streaker location
at1=['Optics 1', 1, 0.0, 220, 15, 0, 90,opt1]
at1A=['Optics 1A', 1, 0.0, 219, 15, 0, 90,opt1B] #backup with lower quad limits
at2=['Optics 2', 2.5, 0.0, 88.5, 15, 0, 90,opt2]
at2A=['Optics 2A', 2.5, 0.0, 88.5, 15, 0, 90,opt2B] #backup with lower quad limits
at3=['Optics 3', 5.0, 0.0, 44.3, 15, 0, 90,opt3]
at3A=['Optics 3', 5.0, 0.0, 44.3, 15, 0, 90,opt3B]  #backup with lower quad limits
at4=['Optics 4', 10.0, 0.0, 22.2, 15, 0, 90,opt4]
at5=['Optics 5', 15.0, 0.0, 14.8, 15, 0, 90,opt5]
at6=['Optics 6', 20.0, 0.0, 11.1, 15, 0, 90,opt6]
at7=['Optics 7', 25.0, 0.0, 8.9, 15, 0, 90,opt7]
at8=['Optics 8', 30.0, 0.0, 7.4, 15, 0, 90,opt8]

quadNames.append('SATBD02.MQUA030')

athos_post_undulator_quads = quadNames
athos_post_undulator_optics = (
        at1,
        at1A,
        at2,
        at2A,
        at3,
        at3A,
        at4,
        at5,
        at6,
        at7,
        at8,
        )
for a in athos_post_undulator_optics:
    a[-1].append(0.)
    a[0] = a[0].replace('Optics', 'SATMA02-UDCP045')

### ATHOS PRE-UNDULATOR

quadNames=['SATDI01.MQUA250', 'SATDI01.MQUA260', 'SATDI01.MQUA280', 'SATDI01.MQUA300', 'SATCB01.MQUA230', 'SATCB01.MQUA430', 'SATCL02.MQUA230', 'SATCL02.MQUA430']

#1.7 m-2 Limits K1
#screen: SATMA01.DSCR030

#Entrance of SATDI01.MQUA250 (Eduard)
#   beta_x=35.76124
#	alpha_x=-1.165823
#	beta_y=24.08687
#	alpha_y=0.6483776

#Transport Quads K1 values:
Fix=[1.017132019081133,-1.7]

AtP1=[0.3650180611825992,-0.7878955467621285,1.699999979846469,-1.699999999999995,1.076009582611750,-1.200354768784456]+Fix
AtP2=[-0.7542168391040225,1.284677569722269,0.4186127069889671,-1.070235617849483,0.4518807347712642,-0.4265873011412286]+Fix
AtP3=[-0.5989575679935852,0.7725074925520736,0.5786288559418530,-0.6530095519272638,0.2307554643234724,-0.2711968944104027]+Fix
AtP4=[-1.159751987924194,0.7828968144880595,0.6685781036264526,-0.7402563591727538,0.8207555214942440,-0.9193960705233104]+Fix

L1=0.15

opt1= [x * L1 for x in AtP1]
opt2= [x * L1 for x in AtP2]
opt3= [x * L1 for x in AtP3]
opt4= [x * L1 for x in AtP4]

#'(Identifier, streaker betax, streaker alphax, streaker betay, streaker alphay, screen betax, screen betay, R11, R12, R33,R34, phase advance x, phase advance y, K1Ls)'
aP1=['Optics 1', 5, -0.026, 5.01, 0.03, 45.67,4.89,-0.148,15.123,0.395,4.599,88.68,68.18,opt1]
aP2=['Optics 2', 9.989, -0.0015, 9.995, -0.0044, 23.10,3.697,-0.148,15.123,0.395,4.599,84.50,49.21,opt2]
aP3=['Optics 3', 19.982, 0.0084, 19.996, 0.0035, 11.92,4.175,-0.148,15.123,0.395,4.599,78.46,30.26,opt3]
aP4=['Optics 4', 29.99, 0.0072, 29.80, 0.0065, 8.317,5.35,-0.148,15.123,0.395,4.599,73.24,21.39,opt4]

athos_pre_undulator_quads = quadNames
athos_pre_undulator_optics = (
        aP1,
        aP2,
        aP3,
        aP4,
        )
for info in athos_pre_undulator_optics:
    info[0] = info[0].replace('Optics', 'SATCL02-UDCP100/200')

